# ScoreMatrix React frontend

A React + Vite frontend for the existing FastAPI backend.

## Requirements
- Node.js LTS and npm
- Existing ScoreMatrix FastAPI backend

## Run locally
1. Extract the project ZIP.
2. Open a terminal in this folder.
3. Install dependencies:
   ```powershell
   npm install
   ```
4. Start the FastAPI backend in its existing folder:
   ```powershell
   uvicorn app.main:app --reload
   ```
5. Start the React frontend in this folder:
   ```powershell
   npm run dev
   ```
6. Open the local URL Vite prints (usually http://localhost:5173).

## Backend CORS
Your backend `.env` should allow `http://localhost:5173` in `CORS_ALLOWED_ORIGINS`. Restart FastAPI after changing `.env`.

## API notes
- Question generation uses the request shape tested in your backend.
- RAG, evaluation, batch evaluation, and plagiarism examples are starter JSON and may need to be edited to match the actual Pydantic request schemas. The app links to `/docs` for each tool.
- Set `VITE_API_BASE` in `.env` if your backend is not at `http://127.0.0.1:8000`.
- This is a local prototype, not production-ready. Add authentication, authorization, and production security before deployment. Never put provider API keys in frontend code.
