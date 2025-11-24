from flask import Flask, render_template, request, jsonify
import os
import io
import time
from google import genai
from google.genai.errors import APIError
from pypdf import PdfReader # Used for PDF text extraction

# --- Environment and Configuration ---

# NOTE 1: This is a Flask backend. You need to run this Python file on a server.
# NOTE 2: The Gemini API Key must be set in your environment variables for this to work.
# Ensure you install the required libraries: pip install Flask google-genai pypdf

# Use os.getenv for standard environment variable access
GEMINI_API_KEY = "AIzaSyAtjk0FV6B3yfKf6VhG2ePSB1iDVLQPDhI"
MODEL_NAME = "gemini-2.0-flash" # The requested Flash model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit for uploads

# --- Gemini Client Initialization (with error handling) ---
client = None
if GEMINI_API_KEY:
    try:
        # Initialize the Gemini Client
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API Client Initialized.")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        client = None
else:
    # A generic message for a missing key
    print("Warning: GEMINI_API_KEY not set. API calls will be disabled.")

# --- Helper Function: PDF Text Extraction ---

def extract_text_from_pdf(pdf_file_stream):
    """Extracts text content from a PDF file stream."""
    try:
        reader = PdfReader(pdf_file_stream)
        text = ""
        for page in reader.pages:
            # We check if page.extract_text() returns something before appending
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

# --- Application Routes ---

@app.route('/')
def index():
    """Renders the main application page."""
    # Flask looks for templates/index.html by default
    return render_template('index.html', title="AI Resume Analyzer")

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """Handles resume upload, calls Gemini API, and returns the analysis."""
    
    # 1. API Key Check
    if client is None:
        return jsonify({"error": "Gemini API client not initialized. Check server logs for API Key error."}), 500

    # 2. Input Validation
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Missing required data: resume file or job description."}), 400

    resume_file = request.files['resume']
    job_description = request.form['job_description']

    if resume_file.filename == '':
        return jsonify({"error": "No resume file selected."}), 400

    if not resume_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

    if not job_description.strip():
        return jsonify({"error": "Job description cannot be empty."}), 400

    try:
        # 3. PDF Parsing
        # Read the file data into an in-memory buffer (BytesIO)
        pdf_stream = io.BytesIO(resume_file.read())
        resume_text = extract_text_from_pdf(pdf_stream)

        if not resume_text:
            return jsonify({"error": "Could not extract readable text from PDF. The document might be an image-only PDF."}), 500
        
        # 4. Prompt Construction
        system_instruction = (
            "You are a world-class Applicant Tracking System (ATS) and Career Coach. "
            "Your task is to analyze a candidate's resume against a specific job description. "
            "Provide the analysis in structured markdown format with four distinct sections. "
            "Ensure the output is clean, well-formatted markdown, ready for display."
        )

        user_prompt = f"""
        Analyze the candidate's resume based on the following job description.

        ## Job Description:
        ---
        {job_description}
        ---

        ## Candidate Resume Text (Extracted):
        ---
        {resume_text}
        ---

        Your analysis must strictly follow this structure:

        1.  **ATS Match Score:** A percentage score out of 100 (e.g., 78/100) indicating the match level. Use bold text for the score.
        2.  **Missing Keywords/Skills:** A bulleted list of up to 5 essential skills/keywords mentioned in the Job Description but *missing* from the Resume. If none are missing, state that clearly.
        3.  **Summary of Strengths:** A concise, single paragraph highlighting the candidate's best relevant qualifications for the role.
        4.  **Actionable Improvement Suggestions:** A bulleted list of 3-5 concrete, specific suggestions on how to modify the resume text to increase the match score.
        """

        # 5. Gemini API Call (with basic retry for robustness)
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Calling Gemini API (Attempt {attempt + 1})...")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                )
                # Successful response
                return jsonify({
                    "success": True,
                    "analysis": response.text
                })

            except APIError as api_e:
                # Handle API errors (e.g., rate limits, invalid key)
                print(f"Gemini API Error on attempt {attempt + 1}: {api_e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    return jsonify({"error": f"Gemini API Error after {MAX_RETRIES} attempts: {str(api_e)}"}), 500
            
            except Exception as e:
                # Handle other unexpected errors during the API call
                print(f"Unexpected error during Gemini API call: {e}")
                return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    except Exception as e:
        # Handle errors during file reading or prompt setup
        print(f"An error occurred during analysis setup: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# --- App Execution ---

if __name__ == '__main__':
    # Instructions to run the app:
    # 1. Ensure you have Flask, google-genai, and pypdf installed:
    #    pip install Flask google-genai pypdf
    # 2. Set your API Key:
    #    export GEMINI_API_KEY="YOUR_API_KEY"
    # 3. Create a directory named 'templates' and save the index.html file there.
    # 4. Run the application:
    #    python app.py
    # 5. Access http://127.0.0.1:5000/ in your browser.
    app.run(debug=True)