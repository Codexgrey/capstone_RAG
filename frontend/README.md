# Frontend — React + TypeScript UI

Patricia's React frontend. Document upload, query input, response display, citations, chat history.

## Setup

```bash
cd frontend
npm install
```

Ensure `frontend/.env` contains:
```env
VITE_API_URL=http://localhost:8000
```

## Run

```bash
npm run dev
```

Opens at http://localhost:5173

## Build for Production

```bash
npm run build
```

## Usage

1. Register or login (admin: `admin@admin.com` / `admin1234`)
2. Click the **Controls** button (top-left hamburger) to open the upload panel
3. Upload PDF, DOCX, TXT, or MD documents
4. In the chat input, select a retrieval method:
   - **Vector** — semantic similarity (FAISS + all-MiniLM-L6-v2)
   - **Keyword** — BM25 keyword search
   - **Hybrid** — FAISS + BM25 fused with Reciprocal Rank Fusion
5. Type a question and press Enter

## Key Components

```
src/
├── components/
│   ├── ChatBox.tsx       # Query input + retrieval method selector
│   ├── ChatHistory.tsx   # Past session list
│   ├── Header.tsx        # Top bar
│   ├── ResponseCard.tsx  # Answer display
│   ├── SideBar.tsx       # Upload panel (all users)
│   └── SourcesPanel.tsx  # Citations + latency + method badge
├── pages/
│   └── App.tsx           # Root layout + auth
└── services/
    ├── apiClient.ts      # Axios instance
    ├── authService.ts    # Login / register / logout
    ├── documentService.ts# Upload, list documents
    └── queryService.ts   # Send query, get sessions
```
