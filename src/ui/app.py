import streamlit as st

def main():
    st.title("I-CAN\nIISc-Conversational-Academic-Navigator")

  
    st.subheader("Ask a question")
    q = st.text_input("Your question:")
    if st.button("Ask") and q:
        st.write("Answer")

if __name__ == "__main__":
    main()
