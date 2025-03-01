import streamlit as st
import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import assemblyai as aai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
import google.auth.exceptions
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Initialize session state
if "vectorstore" not in st.session_state:   
    st.session_state.vectorstore = None
    st.session_state.video_title = None
    st.session_state.transcript = None
    st.session_state.chat_history = []
    st.session_state.summary = None
    st.session_state.quiz = None
    st.session_state.status_message = ""  # New: for displaying current operation status

# API Keys section
st.sidebar.header("API Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
assemblyai_api_key = st.sidebar.text_input("AssemblyAI API Key", type="password")

# Save API keys to environment if provided
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
if assemblyai_api_key:
    os.environ["ASSEMBLYAI_API_KEY"] = assemblyai_api_key
    aai.api_key = assemblyai_api_key

# Function to update status message
def update_status(message):
    st.session_state.status_message = message
    # If we have a status placeholder, update it
    if "status_placeholder" in st.session_state:
        st.session_state.status_placeholder.info(message)

# Initialize Embeddings & LLM (only if API key is provided)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    try:
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)
    except Exception as e:
        st.sidebar.error(f"⚠️ Error initializing embeddings: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=gemini_api_key,
            temperature=0.2,  # Lower temperature for more focused answers
            max_output_tokens=1024  # Ensure adequate response length
        )
    except Exception as e:
        st.sidebar.error(f"⚠️ Error initializing Gemini: {str(e)}")
        return None

# Helper function to extract YouTube video ID
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    return None

# Get YouTube video title using yt-dlp
def get_youtube_title(video_url):
    try:
        update_status("Fetching video title from YouTube...")
        video_id = get_video_id(video_url)
        if not video_id:
            update_status("Failed to extract video ID")
            return "Unknown Title"
            
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = info.get('title', 'Unknown Title')
            return title
    except Exception as e:
        update_status(f"Could not fetch video title: {str(e)}")
        return "Unknown Title"

# Tool: Get YouTube Transcript
def get_youtube_transcript(video_url):
    update_status("Retrieving transcript from YouTube Data API...")
    video_id = get_video_id(video_url)
    if not video_id:
        update_status("Invalid YouTube URL")
        return "Invalid YouTube URL"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript or len(transcript) == 0:
            update_status("Transcript is empty")
            return "Transcript is empty"
        
        # Create a more structured transcript with timestamps
        full_text = ""
        for entry in transcript:
            timestamp = entry["start"]
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            text = entry["text"]
            full_text += f"[{minutes:02d}:{seconds:02d}] {text} "
        
        update_status("Successfully retrieved transcript from YouTube")
        return full_text
    except Exception as e:
        update_status(f"Failed to get transcript from YouTube: {str(e)}")
        return f"Failed to get transcript: {str(e)}"

# Tool: Download YouTube Audio
def download_audio(youtube_url):
    update_status("Downloading audio from YouTube for transcription...")
    try:
        video_id = get_video_id(youtube_url)
        if not video_id:
            update_status("Invalid YouTube URL")
            return "Invalid YouTube URL", None
        audio_dir = "audio"
        os.makedirs(audio_dir, exist_ok=True)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{audio_dir}/{video_id}.%(ext)s",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            update_status("Audio download complete")
            return f"{audio_dir}/{video_id}.{info['ext']}", info.get("title", "Unknown Title")
    except Exception as e:
        update_status(f"Audio download failed: {str(e)}")
        return f"Audio download failed: {str(e)}", None

# Tool: Transcribe Audio with AssemblyAI
def get_transcription(youtube_url):
    with st.spinner("Initiating transcription with AssemblyAI..."):
        if not aai.api_key:
            return "AssemblyAI API key is required for transcription", None

        video_id = get_video_id(youtube_url)
        if not video_id:
            return "Invalid YouTube URL", None

        transcripts_dir = os.path.join(os.getcwd(), "transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)
        transcript_path = os.path.join(transcripts_dir, f"{video_id}.txt")

        # Try to load cached transcript first
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            if not st.session_state.video_title:
                st.session_state.video_title = get_youtube_title(youtube_url)
            return transcript_text, st.session_state.video_title

        # If no cached transcript, download audio and transcribe
        audio_file, video_title = download_audio(youtube_url)
        if isinstance(audio_file, str) and audio_file.startswith("Audio download failed"):
            return audio_file, None

        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            punctuate=True,
            format_text=True
        )
        transcript_obj = transcriber.transcribe(audio_file, config)

        if transcript_obj.text:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_obj.text)
            st.session_state.video_title = video_title
            return transcript_obj.text, video_title

        return "Failed to generate transcript", None

# Tool: Create FAISS Vector Store with improved chunking
def create_vectorstore(text):
    update_status("Creating vector database for semantic search...")
    embeddings = get_embeddings()
    if embeddings is None:
        update_status("Failed to initialize embeddings")
        return "Cannot create vector store: Gemini API key is required"
        
    # Improved chunking strategy for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,        # Larger chunks for more context
        chunk_overlap=200,      # More overlap to preserve context across chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Prioritize natural breaks
    )
    chunks = text_splitter.split_text(text)
    update_status(f"Split transcript into {len(chunks)} chunks")
    
    # Add metadata to chunks for better retrieval
    metadatas = [{"source": "transcript", "chunk": i, "total_chunks": len(chunks)} 
                for i in range(len(chunks))]
    
    vectorstore = FAISS.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        metadatas=metadatas
    )
    st.session_state.vectorstore = vectorstore
    update_status("Vector database initialized successfully")
    return "VectorDB created successfully"

# Tool: Chat with Video using Retrieval (improved)
def chat_with_video(query):
    update_status(f"Processing question: '{query}'...")
    llm = get_llm()
    if llm is None:
        update_status("Gemini API key is required")
        return "Cannot generate answer: Gemini API key is required"
    if not st.session_state.vectorstore:
        update_status("No vector database available")
        return "Please first process a video before asking questions"
    
    try:
        # Get more context by retrieving more documents
        update_status("Retrieving relevant transcript segments...")
        docs = st.session_state.vectorstore.similarity_search(query, k=7)
        update_status(f"Found {len(docs)} relevant segments")
        
        # Extract context with source information
        context_parts = []
        for i, doc in enumerate(docs):
            # Format timestamp information if available
            context = doc.page_content
            metadata = doc.metadata
            context_parts.append(f"Context {i+1}: {context}")
        
        context = "\n\n".join(context_parts)
        
        # Improved prompt with clearer instructions
        update_status("Generating answer based on context...")
        prompt = f"""
You are an AI assistant specialized in answering questions about YouTube videos based on their transcripts. You can also respond in the user's preferred language.

### 🎬 Video Details:  
- **📌 Title:** {st.session_state.video_title}  
- **📜 Context from the video based on the user Query:** {context}  
- **❓ User's Question:** {query}  

### 📌 Instructions:  
1. **🔍 Answer strictly based on the provided context.** No assumptions or extra details.  
2. **⏳ Use timestamps (e.g., 🕒 06:52 - 07:55) if present** to reference specific moments.  
3. **⚠️ If the context lacks enough information, clearly state what is missing.** No guessing.  
4. **✍️ Keep the response clear, engaging, and structured.** Use bullet points and formatting to improve readability.  
5. **🌍 Detect the user's input language from {query} and respond in the same language.**  
6. **💬 Make it feel natural and conversational.** Avoid robotic tone—keep it user-friendly!  
7. **🎭 Add emojis based on the context.** For example:  
   - 🎮 For gaming-related content  
   - 🎤 For interviews or discussions  
   - ⚔️ For intense or competitive moments  
   - 😂 For humorous or lighthearted sections  

### ✅ Your Answer:
"""
        response = llm.invoke(prompt)
        update_status("Answer generated successfully")
        return response.content
    except Exception as e:
        update_status(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

# Tool: Summarize YouTube Video (improved) /// https://github.com/DevRico003/youtube_summarizer
def summarize_video():
    update_status("Generating video summary...")
    if "transcript" not in st.session_state or not st.session_state.transcript:
        update_status("No transcript available")
        return "Video not processed or transcript missing."
    
    if st.session_state.summary:
        update_status("Returning cached summary")
        return st.session_state.summary
        
    llm = get_llm()
    if llm is None:
        update_status("LLM not available")
        return "LLM is not available."
        
    prompt = f"""
You are an expert at summarizing YouTube videos.

Video Title: {st.session_state.video_title or 'Unknown'}

Transcript: {st.session_state.transcript[:50000]}
Instructions:
1. Provide a concise, well-structured summary of the video (100 words).
2. Begin with a one-sentence overview of what the video is about.
3. Identify and summarize 3-5 key points or topics discussed in the video.
4. Include any important conclusions or takeaways.
5. Organize the summary with clear paragraph breaks and bullet points where appropriate.
6. If timestamps are available in the transcript ([MM:SS]), include the most important ones.

Your summary:
"""
    response = llm.invoke(prompt)
    summary = response.content
    st.session_state.summary = summary
    update_status("Summary generated successfully")
    return summary

# Tool: Generate Quiz from YouTube Video (improved)
def generate_quiz():
    update_status("Generating quiz from video content...")
    if "transcript" not in st.session_state or not st.session_state.transcript:
        update_status("No transcript available")
        return "Video not processed or transcript missing."
        
    if st.session_state.quiz:
        update_status("Returning cached quiz")
        return st.session_state.quiz
        
    llm = get_llm()
    if llm is None:
        update_status("LLM not available")
        return "LLM is not available."
        
    prompt = f"""
You are an expert educator. Based on the following YouTube video transcript and title, please generate a five-question multiple choice quiz.

Video Title: {st.session_state.video_title or 'Unknown'}
Transcript: {st.session_state.transcript[:50000]}  # Use first 50K characters if transcript is too long

Instructions:
1. Create 5 meaningful multiple-choice questions that test understanding of key concepts from the video.
2. For each question, provide 4 options (A, B, C, D) with only one correct answer.
3. Ensure questions vary in difficulty from basic recall to critical thinking.
4. After all questions, provide the correct answers in a separate answer key.
5. Format the quiz clearly with proper spacing and numbering.
6. Ensure all questions are focused on the most important points from the video.

Your quiz:
"""
    response = llm.invoke(prompt)
    quiz = response.content
    st.session_state.quiz = quiz
    update_status("Quiz generated successfully")
    return quiz

# Intelligent Transcript Acquisition Logic
def smart_get_transcript(url):
    """Intelligent function that tries multiple methods to get a transcript"""
    with st.spinner("Attempting to retrieve transcript from YouTube API..."):
        transcript = get_youtube_transcript(url)
    if transcript and not transcript.startswith("Failed") and not transcript.startswith("Invalid") and not transcript.startswith("Transcript is empty"):
        return transcript, get_youtube_title(url)

    with st.spinner("YouTube API failed. Falling back to AssemblyAI transcription..."):
        transcript, title = get_transcription(url)
    if transcript and not transcript.startswith("Failed") and not transcript.startswith("Audio download failed"):
        return transcript, title

    return None, None

# Direct video processing function with intelligent transcript retrieval
def process_video(url):
    # Reset status message
    st.session_state.status_message = ""
    
    update_status(f"Processing video: {url}")
    st.session_state.video_title = get_youtube_title(url)
    
    # Use our intelligent transcript retrieval function
    transcript, title = smart_get_transcript(url)
    # Update title if available
    if title:
        st.session_state.video_title = title
    
    if transcript:
        st.session_state.transcript = transcript  # save for summary/quiz later
        create_vectorstore(transcript)
        update_status(f"Video processing complete: {st.session_state.video_title}")
        # Reset summary and quiz
        st.session_state.summary = None
        st.session_state.quiz = None
        return True
    else:
        update_status("Failed to process video - could not obtain transcript")
        return False

# Define Tools for Agent
def get_tools():
    return [
        Tool(
            name="get_youtube_transcript", 
            func=get_youtube_transcript, 
            description="Fetches YouTube video transcript given a YouTube URL using the YouTube Transcript API."
        ),
        Tool(
            name="download_audio", 
            func=download_audio, 
            description="Downloads audio from a YouTube video given a YouTube URL when the transcript is not available."
        ),
        Tool(
            name="get_transcription", 
            func=get_transcription, 
            description="Transcribes YouTube audio using AssemblyAI if direct transcript fetch fails. Returns transcript text and video title."
        ),
        Tool(
            name="smart_get_transcript", 
            func=smart_get_transcript, 
            description="Intelligent function that tries multiple methods to get a transcript, starting with YouTube API and falling back to AssemblyAI."
        ),
        Tool(
            name="create_vectorstore", 
            func=create_vectorstore, 
            description="Creates a FAISS vectorstore from the transcript text for enabling Q&A."
        ),
        Tool(
            name="chat_with_video", 
            func=chat_with_video, 
            description="Answers questions about the YouTube video based on its transcript."
        ),
        Tool(
            name="summarize_video", 
            func=summarize_video, 
            description="Generates a concise summary of the YouTube video using its transcript."
        ),
        Tool(
            name="generate_quiz", 
            func=generate_quiz, 
            description="Creates a five-question multiple choice quiz based on the YouTube video transcript."
        )
    ]

# Create a tool-calling agent with an updated prompt
def get_agent():
    llm = get_llm()
    if llm is None:
        return None
    tools = get_tools()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            """You are an intelligent AI assistant for analyzing YouTube videos. Your primary goal is to help users understand video content through accurate Q&A, summarization, and quiz generation.

Your process for handling YouTube videos should ALWAYS follow this exact sequence:

1. FIRST, try to get the transcript directly from YouTube using the YouTube Transcript API.
   - Use the get_youtube_transcript tool for this purpose.
   - This is the preferred method as it's faster and more accurate.

2. ONLY IF the YouTube Transcript API fails, fall back to transcription with AssemblyAI:
   - Use download_audio to get the audio file
   - Then use get_transcription to transcribe with AssemblyAI

3. Create a vector database from the transcript for semantic search.

When answering questions:
- Use only information present in the transcript
- Reference timestamps when available
- Be direct and factual in your responses
- Admit when you don't have enough information

For a new video, ALWAYS use smart_get_transcript which implements this intelligent logic automatically.
"""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent

# Enhanced agent executor for Q&A and tool usage
def ask_agent(question):
    update_status(f"Processing your request: '{question}'...")
    agent = get_agent()
    if agent is None:
        update_status("Cannot initialize agent: Gemini API key is required")
        return "Cannot initialize agent: Gemini API key is required"
    
    try:
        inputs = {
            "input": question,
            "chat_history": st.session_state.chat_history
        }
        update_status("Working on your request...")
        result = agent.invoke(inputs)
        output = result.get("output", "I couldn't generate a response.")
        
        st.session_state.chat_history.append({"role": "human", "content": question})
        st.session_state.chat_history.append({"role": "ai", "content": output})
        
        update_status("Request completed successfully")
        return output
    except Exception as e:
        update_status(f"Error using agent: {str(e)}")
        return f"Error using agent: {str(e)}"

# Streamlit UI with tabs for different functions and spinners for progress
st.title("🎬 YouTube Video Analysis & Q&A")
DEFAULT_WIDTH = 80
# Check if API keys are provided
if not gemini_api_key:
    st.warning("⚠️ Please enter a Gemini API key in the sidebar to use this application")

if not assemblyai_api_key:
    st.warning("⚠️ Please enter an AssemblyAI API key in the sidebar for audio transcription")

# Show current status in the sidebar (left side)
st.sidebar.subheader("Current Status")
if st.session_state.status_message:
    st.sidebar.info(st.session_state.status_message)
else:
    st.sidebar.write("Ready to process video")

# Only show the main UI if API keys are provided
if gemini_api_key and assemblyai_api_key:
    # Main UI in a single column
    st.subheader("Enter a YouTube Video")
    video_url = st.text_input("Paste YouTube URL:")

    if video_url:
        st.video(video_url)
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                success = process_video(video_url)
            if success:
                st.success(f"Video processed successfully: {st.session_state.video_title}")
            else:
                st.error("Failed to process video. Could not obtain transcript.")
    
    # If a video has been processed, show the analysis tabs
    if st.session_state.vectorstore is not None:
        st.subheader(f"Analysis for: {st.session_state.video_title}")

        tab1, tab2, tab3 = st.tabs(["Q&A", "Summary", "Quiz"])
        
        with tab1:
            st.subheader("Ask Questions About the Video")
            user_question = st.text_input("Your question about the video:")
            if user_question:
                with st.spinner("Searching transcript and formulating answer..."):
                    response = chat_with_video(user_question)
                st.markdown("### Answer:")
                st.markdown(response)
                
            if st.session_state.chat_history:
                with st.expander("View Chat History", expanded=False):
                    for message in st.session_state.chat_history:
                        role = message["role"]
                        content = message["content"]
                        if role == "human":
                            st.markdown(f"**You:** {content}")
                        else:
                            st.markdown(f"**AI:** {content}")
                            st.divider()

        with tab2:
            st.subheader("Video Summary")
            if st.button("Generate Summary"):
                with st.spinner("Analyzing video content and creating summary..."):
                    summary = summarize_video()
                st.markdown(summary)
            elif st.session_state.summary:
                st.markdown(st.session_state.summary)

        with tab3:
            st.subheader("Knowledge Quiz")
            if st.button("Generate Quiz"):
                with st.spinner("Creating educational quiz based on video content..."):
                    quiz = generate_quiz()
                st.markdown(quiz)
            elif st.session_state.quiz:
                st.markdown(st.session_state.quiz)
else:
    st.info("Enter the required API keys in the sidebar to get started")
