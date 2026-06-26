import os
import re
import requests
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")

KEYWORD_MAP = {
    "developer":  ["developer", "engineer", "programmer", "coder"],
    "engineer":   ["engineer", "developer", "architect", "programmer"],
    "ai":         ["ai", "machine learning", "ml", "data scientist", "llm", "nlp", "generative", "deep learning"],
    "web":        ["frontend", "front-end", "react", "angular", "vue", "full stack", "web developer"],
    "backend":    ["backend", "back-end", "server", "api", "django", "flask", "node"],
    "frontend":   ["frontend", "front-end", "react", "angular", "vue", "ui developer"],
    "python":     ["python", "django", "flask", "fastapi", "data scientist"],
    "java":       ["java", "spring", "kotlin", "jvm"],
    "data":       ["data scientist", "data engineer", "data analyst", "ml engineer", "analytics"],
    "devops":     ["devops", "cloud", "aws", "azure", "gcp", "infrastructure", "sre", "kubernetes"],
    "mobile":     ["mobile", "android", "ios", "react native", "flutter", "swift"],
    "security":   ["security", "cybersecurity", "penetration", "infosec", "soc analyst"],
    "fullstack":  ["full stack", "fullstack", "full-stack", "mern", "mean"],
    "react":      ["react", "frontend", "front-end", "javascript", "next.js"],
    "cloud":      ["cloud", "aws", "azure", "gcp", "devops", "infrastructure"],
}

REMOTE_KEYWORDS = [
    "remote", "work from home", "wfh", "anywhere",
    "distributed", "fully remote", "100% remote"
]

def get_search_terms(keyword):
    kw = keyword.lower().strip()
    terms = set([kw])
    for key, synonyms in KEYWORD_MAP.items():
        if kw == key or kw in synonyms:
            terms.update(synonyms)
    return list(terms)

def matches_search(job_title, job_description, terms):
    title = job_title.lower()
    desc  = job_description.lower()[:300]
    score = 0
    for term in terms:
        if term in title:
            score += 3
        elif term in desc:
            score += 1
    return score

def is_remote(job):
    # only check title and location, NOT description
    # description uses "remote" as a technical term too often
    text = " ".join([
        job.get("title", ""),
        job.get("location", ""),
        job.get("remote", ""),
    ]).lower()
    return any(kw in text for kw in REMOTE_KEYWORDS)

def clean_html(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_job(title, description):
    try:
        message = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": f"One sentence job summary:\nTitle: {title}\nDesc: {description[:200]}"
            }]
        )
        return message.choices[0].message.content.strip()
    except:
        return "Click to view full job details."

def normalize_job(job):
    return {
        "title":       job.get("title", "Unknown Title"),
        "company":     job.get("company", "Unknown Company"),
        "location":    job.get("location", ""),
        "country":     job.get("country", ""),
        "url":         job.get("url", "#"),
        "salary":      job.get("salary", ""),
        "job_type":    job.get("job_type", ""),
        "remote":      "Remote" if is_remote(job) else job.get("remote", ""),
        "posted_date": job.get("posted_date", ""),
        "source":      job.get("source", ""),
        "tags":        job.get("tags", [])[:6],
        "description": job.get("description", "")[:2000],
        "_score":      job.get("_score", 0),
    }

def fetch_remotive(keyword):
    try:
        res = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"category": "software-dev"},
            timeout=8
        )
        data = res.json()
        jobs = []
        for j in data.get("jobs", []):
            jobs.append({
                "title":       j.get("title", ""),
                "company":     j.get("company_name", ""),
                "location":    j.get("candidate_required_location", "Worldwide"),
                "country":     "Worldwide",
                "url":         j.get("url", "#"),
                "salary":      j.get("salary", ""),
                "job_type":    j.get("job_type", "full_time"),
                "remote":      "Remote",
                "posted_date": j.get("publication_date", "")[:10],
                "source":      "Remotive",
                "tags":        j.get("tags", []),
                "description": clean_html(j.get("description", "")),
            })
        return jobs
    except:
        return []

def fetch_adzuna(keyword, country_code):
    try:
        url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
        res = requests.get(url, params={
            "app_id":           ADZUNA_APP_ID,
            "app_key":          ADZUNA_API_KEY,
            "what":             keyword,
            "results_per_page": 30,
            "content-type":     "application/json"
        }, timeout=8)
        data = res.json()
        jobs = []
        for j in data.get("results", []):
            loc = j.get("location", {})
            location_parts = loc.get("display_name", "") if isinstance(loc, dict) else ""
            salary_min = j.get("salary_min")
            salary_max = j.get("salary_max")
            salary = ""
            if salary_min and salary_max:
                salary = f"${int(salary_min):,} - ${int(salary_max):,}"
            elif salary_min:
                salary = f"From ${int(salary_min):,}"
            jobs.append({
                "title":       j.get("title", ""),
                "company":     j.get("company", {}).get("display_name", ""),
                "location":    location_parts,
                "country":     country_code.upper(),
                "url":         j.get("redirect_url", "#"),
                "salary":      salary,
                "job_type":    j.get("contract_type", "full_time"),
                "remote":      "",
                "posted_date": j.get("created", "")[:10],
                "source":      "Adzuna",
                "tags":        [j.get("category", {}).get("label", "")],
                "description": clean_html(j.get("description", "")),
            })
        return jobs
    except:
        return []

def deduplicate(jobs):
    seen = set()
    unique = []
    for job in jobs:
        key = (job["title"].lower().strip(), job["company"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique

@app.route("/")
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.route("/search")
def search():
    keyword  = request.args.get("keyword", "").strip()
    page     = int(request.args.get("page", 1))
    country  = request.args.get("country", "all")
    job_type = request.args.get("job_type", "").lower()
    remote   = request.args.get("remote", "").lower()
    per_page = 6

    if not keyword:
        return jsonify({"jobs": [], "total": 0, "page": 1, "has_more": False})

    search_terms = get_search_terms(keyword)

    # for "all countries" only query top 3 to keep it fast
    if country == "all":
        adzuna_countries = ["in", "gb", "us"]
    else:
        adzuna_countries = [country]

    futures_map = {}
    executor = ThreadPoolExecutor(max_workers=5)
    futures_map[executor.submit(fetch_remotive, keyword)] = "remotive"
    for c in adzuna_countries:
        futures_map[executor.submit(fetch_adzuna, keyword, c)] = f"adzuna_{c}"

    all_jobs = []
    for future in as_completed(futures_map):
        try:
            all_jobs.extend(future.result())
        except:
            pass
    executor.shutdown(wait=False)

    # score
    scored = []
    for job in all_jobs:
        score = matches_search(job["title"], job["description"], search_terms)
        if score > 0:
            job["_score"] = score
            scored.append(job)

    scored = deduplicate(scored)

    # apply filters
    if job_type:
        scored = [j for j in scored if job_type in (j.get("job_type") or "").lower()]

    if remote == "remote":
        scored = [j for j in scored if is_remote(j)]

    scored.sort(key=lambda j: -j.get("_score", 0))

    total = len(scored)
    start = (page - 1) * per_page
    end   = start + per_page
    paged = scored[start:end]

    normalized = [normalize_job(j) for j in paged]

    def add_summary(job):
        job["summary"] = summarize_job(job["title"], job["description"])
        return job

    with ThreadPoolExecutor(max_workers=6) as ex:
        jobs_out = list(ex.map(add_summary, normalized))

    return jsonify({
        "jobs":     jobs_out,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "has_more": end < total
    })

@app.route("/suggestions")
def suggestions():
    query = request.args.get("q", "").lower()
    all_suggestions = [
        "Software Engineer", "Python Developer", "Frontend Developer",
        "Backend Developer", "Full Stack Developer", "React Developer",
        "Java Developer", "DevOps Engineer", "Machine Learning Engineer",
        "Data Scientist", "Cloud Engineer", "Cybersecurity Analyst",
        "Mobile Developer", "iOS Developer", "Android Developer",
        "UI/UX Designer", "Product Manager", "QA Engineer",
        "Node.js Developer", "Django Developer", "AI Engineer",
        "NLP Engineer", "Data Engineer", "Site Reliability Engineer",
        "Blockchain Developer", "Flutter Developer", "Golang Developer",
    ]
    if not query:
        return jsonify([])
    matches = [s for s in all_suggestions if query in s.lower()]
    return jsonify(matches[:6])

if __name__ == "__main__":
    app.run(debug=True)