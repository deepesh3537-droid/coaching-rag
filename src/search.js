import "dotenv/config";

import { MongoClient } from "mongodb";
import { pipeline } from "@huggingface/transformers";

const client = new MongoClient(process.env.MONGODB_URI);

const question = "What is the fee for the one-year JEE Main course?";

try {
  // 1. Embed the question using the SAME model
  console.log("1. Script started");
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
   console.log("2. Embedding model loaded");
  const result = await extractor(question, {
    pooling: "mean",
    normalize: true,
  });
  console.log("3. Question embedded");

  const queryVector = result.tolist()[0];

  // 2. Search MongoDB
  await client.connect();
  console.log("4. MongoDB connected");

  const collection = client
    .db("coaching_rag")
    .collection("document_chunks");

  const results = await collection.aggregate([
    {
      $vectorSearch: {
        index: "vector_index",
        path: "embedding",
        queryVector,
        numCandidates: 100,
        limit: 4,
        filter: {
          documentId: "northstar-prospectus"
        }
      }
    },
    {
      $project: {
        _id: 0,
        text: 1,
        metadata: 1,
        score: { $meta: "vectorSearchScore" }
      }
    }
  ]).toArray();
  console.log("5. Vector search completed");
console.log("Documents found:", results.length);


  // 3. Inspect retrieved passages
  for (const [index, result] of results.entries()) {
    console.log(`\n--- Result ${index + 1} ---`);
    console.log("Score:", result.score);
    console.log("Metadata:", result.metadata);
    console.log("Text:", result.text);
  }

  // 4. Generate the answer using Ollama

  if (results.length === 0) {
    console.log("No relevant passages found.");
  } else {
    const context = results
      .map((item) => item.text)
      .join("\n\n");

    const response = await fetch(
      "http://localhost:11434/api/chat",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "qwen2.5:1.5b",
          stream: false,
          messages: [
            {
              role: "system",
              content:
                "Answer using only the supplied context. " +
                "Do not invent facts. If the answer is missing, " +
                "say the document does not contain enough information.",
            },
            {
              role: "user",
              content: `Context:\n${context}\n\nQuestion: ${question}`,
            },
          ],
          options: {
            temperature: 0.1,
            num_predict: 200,
          },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.status}`);
    }

    const data = await response.json();

    console.log("\n--- Final Answer ---");
    console.log(data.message.content);
  }


} catch (error) {
  console.error("Search failed:", error);
} finally {
  await client.close();
}