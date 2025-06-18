import os
import smtplib
from apscheduler.schedulers.blocking import BlockingScheduler
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

# ---- Load environment variables ----
load_dotenv()

os.environ["OPENAI_API_KEY"] = ""

EMAIL_USER = os.getenv("EMAIL_USER","")
EMAIL_PASS = os.getenv("EMAIL_PASS","")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TO_EMAIL = ""
INDEX_PATH = "src/data/faiss_index"

# ---- OpenAI Client ----
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Functions ----

def load_vector_store():
    #embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def get_recent_documents(vectorstore, query="weekly update", k=5):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def generate_weekly_digest(text_block):
    prompt = f"""
You are an assistant that summarizes academic or conversation data.
Summarize the following information into a short, clear weekly digest:

{text_block}

Focus on clarity, relevance, and concise formatting.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content


def send_digest_email(digest_body: str):
    subject = f"📬 Weekly Digest - {datetime.now().strftime('%Y-%m-%d')}"

    # Build HTML template
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2>📬 IISc Weekly Digest</h2>
        <p>Here are your academic updates for the week:</p>
        <ul>
    """

    # Convert plain digest into HTML bullet points
    for line in digest_body.strip().split("\n"):
        if line.strip():  # skip blank lines
            html_content += f"<li>{line.strip()}</li>"

    html_content += """
        </ul>
        <p style="margin-top: 30px;">Stay organized and have a great week! 💡</p>
    </body>
    </html>
    """

    # Prepare the email
    msg = MIMEMultipart("alternative")
    msg['From'] = EMAIL_USER
    msg['To'] = TO_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(html_content, 'html'))

    # Send
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        print(f"[{datetime.now()}] Digest email sent!")
    except Exception as e:
        print(f" [{datetime.now()}] Email sending failed:", e)


def run_digest_job():
    print(f"⏳ [{datetime.now()}] Running weekly digest job...")
    vectorstore = load_vector_store()
    content = get_recent_documents(vectorstore)

    if not content.strip():
        print("⚠️ No content found in vector store.")
        return

    digest = generate_weekly_digest(content)
    send_digest_email(digest)

# ---- Scheduler ----

scheduler = BlockingScheduler()

@scheduler.scheduled_job('interval', minutes=1)  # change to 'cron' for weekly , using minute for testing
def scheduled_task():
    run_digest_job()

if __name__ == "__main__":
    print(" Scheduler running... (every 1 minute)")
    scheduler.start()
