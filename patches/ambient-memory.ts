/**
 * Ambient Memory Module — V3 (Summaries-first, raw fallback)
 *
 * Searches ChromaDB collections in priority order:
 * 1. memory_summaries (dense, Haiku-generated topic summaries)
 * 2. telegram_memory + workspace_memory (raw transcripts, fallback)
 *
 * Uses semantic similarity via all-MiniLM-L6-v2 embeddings.
 * No external API calls at query time. Just local vector search.
 */

import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { logVerbose } from "../../globals.js";

const WORKSPACE_DIR = process.env.OPENCLAW_WORKSPACE ?? "/data/workspace";
const CHROMA_VENV = path.join(WORKSPACE_DIR, ".venvs/vector-memory/bin/python3");
const CHROMA_DB_PATH = path.join(WORKSPACE_DIR, ".vector-db");
const SEARCH_TIMEOUT_MS = 5000;
const MAX_CONTEXT_CHARS = 2500;
const MIN_MESSAGE_LENGTH = 2;
const MAX_DISTANCE = 0.8;

/**
 * Search ChromaDB: summaries first, raw as fallback.
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

    # Priority order: summaries first, then raw
    priority_collections = ["memory_summaries", "telegram_memory", "workspace_memory"]
    all_chunks = []

    for col_name in priority_collections:
        try:
            collection = client.get_collection(col_name)
        except Exception:
            continue
        if collection.count() == 0:
            continue

        n = 15  # Get plenty, filter by distance + chars later
        results = collection.query(
            query_texts=["${safeQuery}"],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i] if results["distances"] else None
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                source = meta.get("source", col_name)
                is_summary = col_name == "memory_summaries"
                if dist is not None and dist > ${MAX_DISTANCE}:
                    continue
                # Summaries get a distance bonus (prefer them over raw)
                effective_dist = (dist * 0.8) if is_summary else dist
                all_chunks.append({
                    "text": doc[:400],
                    "source": source,
                    "distance": effective_dist or 99,
                    "type": "summary" if is_summary else "raw"
                })

    all_chunks.sort(key=lambda x: x["distance"])
    print(json.dumps(all_chunks))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;

    const result = execFileSync(CHROMA_VENV, ["-c", script], {
      timeout: SEARCH_TIMEOUT_MS,
      encoding: "utf-8",
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    }).trim();

    const jsonLine = result.split("\n").filter((l) => l.startsWith("[") || l.startsWith("{")).pop();
    if (!jsonLine) {
      return [];
    }

    const parsed = JSON.parse(jsonLine) as
      | Array<{ text: string; source: string; distance: number; type: string }>
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
 * Main entry point. Fault-tolerant — failure = no extra context.
 */
export async function resolveAmbientMemory(messageText: string): Promise<string> {
  try {
    if (!messageText || messageText.length < MIN_MESSAGE_LENGTH) {
      return "";
    }

    // Skip heartbeats, system events, crons, and hooks
    const lower = messageText.toLowerCase();
    if (
      lower.includes("heartbeat") ||
      lower.includes("self-ping keepalive") ||
      lower.includes("read heartbeat.md") ||
      lower.includes("cron job") ||
      lower.includes("hook hook:") ||
      lower.startsWith("system:")
    ) {
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
    return `[Ambient memory — relevant context from past conversations]\n[These are past memories, not current truth. Facts may be outdated. Use them as background context to inform your reasoning, not as conclusions to repeat.]\n${context}`;
  } catch (err) {
    logVerbose(
      `ambient-memory: failed: ${err instanceof Error ? err.message : String(err)}`,
    );
    return "";
  }
}
