import streamlit as st
from quiz import quiz

def main():
    st.title("House Robber II Quiz 🏠💰")
    main_quiz = quiz["quiz"]
    
    if "score" not in st.session_state:
        st.session_state.score = 0
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    
    current_index = st.session_state.current_question

    if current_index < len(main_quiz):
        question_data = main_quiz[current_index]
        st.subheader(f"Question {current_index + 1}")
        st.write(question_data["question"])

        if f"user_answer_{current_index}" not in st.session_state:
            st.session_state[f"user_answer_{current_index}"] = None
        
        user_answer = st.radio("Choose your answer:", question_data["options"],
                               key=f"q{current_index}", index=None)
        
        if st.button("Submit"):
            if user_answer is not None:
                if user_answer == question_data["answer"]:
                    st.session_state.score += 1
                    st.success("Correct! ✅")
                else:
                    st.error(f"Wrong! ❌ The correct answer is {question_data['answer']}")

                st.session_state.current_question += 1
                st.rerun()
            else:
                st.warning("Please select an answer before submitting!")
    else:
        st.write("## Quiz Completed! 🎉")
        st.write(f"Your final score: {st.session_state.score}/{len(main_quiz)}")
        
        if st.button("Restart Quiz"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
