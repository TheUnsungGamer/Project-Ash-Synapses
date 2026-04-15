# Projecy-Ash-Synapses
Interactive Neural Visualization of Conversation Memory
Synapses is a specialized engine within the Project Ash ecosystem designed to transform flat, chronological chat logs into a living, multi-dimensional semantic network. By leveraging vector embeddings and force-directed graph theory, it maps the "synaptic" connections between disparate ideas, projects, and timestamps.

🛠 Core Tech Stack
Vector Engine: OpenAI / HuggingFace Embeddings for high-dimensional semantic mapping.

Graph Logic: Force-directed graph algorithms (D3.js or Vis.js) for spatial relationship modeling.

Backend: FastAPI / Python for rapid embedding generation and retrieval.

Visualization: Interactive 2D/3D Canvas rendering.

🚀 Key Features
Semantic Clustering: Automatically groups related topics (e.g., coding snippets, baking formulas, infrastructure strategy) regardless of when they were discussed.

Temporal Decay Visualization: Nodes representing older memories can be configured to "dim" or drift to the periphery, simulating human forgetfulness or shifting focus.

Cross-Link Discovery: Identifies hidden "synapses" between projects where similar logic or keywords overlap.

Real-time Expansion: Dynamically injects new conversation data into the existing graph without a full re-index.

📂 Project Structure
├── data/               # Raw and processed JSON chat logs (Gitignored)
├── src/
│   ├── embeddings/     # Logic for vectorization and semantic analysis
│   ├── graph/          # Force-directed layout and relationship logic
│   └── web/            # Dashboard and visualization front-end
├── .gitignore          # Essential security for API keys and local indices
└── requirements.txt    # Project dependencies
⚡ Quick Start
Clone the repository:

Bash
git clone https://github.com/TheUnsungGamer/Projecy-Ash-Synapses.git
Install Dependencies:

Bash
pip install -r requirements.txt
Environment Setup:
Create a .env file and add your API keys (see .env.example).

Run Visualization:

Bash
python main.py
📜 Technical Philosophy
Unlike traditional RAG (Retrieval-Augmented Generation) systems that focus on finding a single answer, Synapses focuses on contextual architecture. It is designed for the "Grounding" of AI agents, giving them a persistent, visualizable map of a user's entire intellectual history.
