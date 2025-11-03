import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job_description = """We are looking for a software engineer with experience in Python, machine learning, and automation. Strong understanding of AI workflows and data processing preferred."""

resumes = {
    "Drea Resume 1": "Experienced in recruiting, data analytics, and process automation using Python and SQL. Currently studying Computer Science with a focus on AI.",
    "Resume 2": "Software engineer with 5 years of experience in React and JavaScript development. Interested in front-end frameworks and UX design.",
    "Resume 3": "Python developer with a passion for automation, APIs, and machine learning. Built multiple automation tools and AI-powered dashboards.",
}

def match_resume(job_text, resume_dict):
    texts = [job_text] + list(resume_dict.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    results = sorted(zip(resume_dict.keys(), similarities), key=lambda x: x[1], reverse=True)
    return "\n".join([f"{name}: {round(score*100, 2)}% match" for name, score in results])

def run_matcher(job_desc, res1, res2, res3):
    resumes_input = {
        "Custom Resume 1": res1,
        "Custom Resume 2": res2,
        "Custom Resume 3": res3,
    }
    return match_resume(job_desc, resumes_input)

with gr.Blocks() as demo:
    gr.Markdown("## 💼 MatchMaker.AI — Resume Match Engine\nMatch your job descriptions with resumes using smart AI scoring.")
    with gr.Row():
        job_input = gr.Textbox(label="📝 Job Description", lines=6, value=job_description)
    with gr.Row():
        resume1 = gr.Textbox(label="📄 Resume 1", lines=4, value=list(resumes.values())[0])
        resume2 = gr.Textbox(label="📄 Resume 2", lines=4, value=list(resumes.values())[1])
        resume3 = gr.Textbox(label="📄 Resume 3", lines=4, value=list(resumes.values())[2])
    match_btn = gr.Button("🔍 Match Resumes")
    output = gr.Textbox(label="📊 Match Results", lines=6)
    match_btn.click(fn=run_matcher, inputs=[job_input, resume1, resume2, resume3], outputs=output)
demo.launch()
