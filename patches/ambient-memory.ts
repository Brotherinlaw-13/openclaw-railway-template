/**
 * Ambient Memory Module — V2 (No LLM, direct semantic search)
 *
 * Passes the incoming message directly to ChromaDB as a query.
 * ChromaDB's all-MiniLM-L6-v2 embedding model handles semantic similarity natively.
 * No Haiku, no API key, no external dependencies. Just local vector search.
 *
 * Searches both telegram_memory and workspace_memory collections in .vector-db/
 */

import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { logVerbose } from "../../globals.js";

const WORKSPACE_DIR = process.env.OPENCLAW_WORKSPACE ?? "/data/workspace";
const CHROMA_VENV = path.join(WORKSPACE_DIR, ".venvs/vector-memory/bin/python3");
const CHROMA_DB_PATH = path.join(WORKSPACE_DIR, ".vector-db");
const SEARCH_TIMEOUT_MS = 5000;
const MAX_CONTEXT_CHARS = 2000;
const MIN_MESSAGE_LENGTH = 2;

/**
 * Search ChromaDB for relevant memory chunks using the raw message as query.
 */
function searchMemory(messageText: string): string[] {
  const query = messageText
    .replace(/^\[.*?\]\s*/g, "")
    .replace(/^System:\s*/gi, "")
    .trim();

  if (!query || query.length < MIN_MESSAGE_LENGTH) {
    return [];
  }

  try {
    const safeQuery = query
      .replace(/\\/g, "\\\\")
      .replace(/"/g, '\\"')
      .replace(/\n/g, "\\n")
      .replace(/\r/g, "\\r");

    const script = `
import json, sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import chromadb
    client = chromadb.PersistentClient(path="${CHROMA_DB_PATH}")
    all_chunks = []
    for col_name in ["telegram_memory", "workspace_memory"]:
        try:
            collection = client.get_collection(col_name)
        except Exception:
            continue
        if collection.count() == 0:
            continue
        results = collection.query(
            query_texts=["${safeQuery}"],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i] if results["distances"] else None
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                source = meta.get("source", col_name)
                if dist is not None and dist > 1.5:
                    continue
                all_chunks.append({"text": doc[:400], "source": source, "distance": dist or 99})
    all_chunks.sort(key=lambda x: x["distance"])
    print(json.dumps(all_chunks[:5]))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;

    const result = execFileSync(CHROMA_VENV, ["-c", script], {
      timeout: SEARCH_TIMEOUT_MS,
      encoding: "utf-8",
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    }).trim();

    // Filter out progress bar lines from chromadb/onnx downloads
    const jsonLine = result.split("\n").filter((l) => l.startsWith("[") || l.startsWith("{")).pop();
    if (!jsonLine) {
      return [];
    }

    const parsed = JSON.parse(jsonLine) as
      | Array<{ text: string; source: string; distance: number }>
      | { error: string };

    if ("error" in parsed) {
      logVerbose(`ambient-memory: ChromaDB error: ${parsed.error}`);
      return [];
    }

    if (!Array.isArray(parsed) || parsed.length === 0) {
      return [];
    }

    return parsed.map((chunk) => `[${chunk.source}] ${chunk.text}`);
  } catch (err) {
    logVerbose(
      `ambient-memory: memory search failed: ${err instanceof Error ? err.message : String(err)}`,
    );
    return [];
  }
}

/**
 * Main entry point: search ChromaDB with the message text, return context block.
 * Returns empty string if no relevant context found.
 * Fault-tolerant — failure just means no extra context.
 */
export async function resolveAmbientMemory(messageText: string): Promise<string> {
  try {
    if (!messageText || messageText.length < MIN_MESSAGE_LENGTH) {
      return "";
    }

    const chunks = searchMemory(messageText);
    if (chunks.length === 0) {
      return "";
    }

    let context = "";
    for (const chunk of chunks) {
      if (context.length + chunk.length + 2 > MAX_CONTEXT_CHARS) {
        break;
      }
      context += (context ? "\n" : "") + chunk;
    }

    if (!context) {
      return "";
    }

    logVerbose(`ambient-memory: injecting ${context.length} chars of context`);
    return `[Ambient memory — relevant context from past conversations]\n${context}`;
  } catch (err) {
    logVerbose(
      `ambient-memory: failed: ${err instanceof Error ? err.message : String(err)}`,
    );
    return "";
  }
}
