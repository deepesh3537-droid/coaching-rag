import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { MongoClient } from "mongodb";
import { pipeline } from "@huggingface/transformers";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "dotenv/config"

const client = new MongoClient(process.env.MONGODB_URI)

const DOCUMENT_ID = "northstar-prospectus"

//load the doc
const loader = new PDFLoader("./documents/fees.pdf")

const documents = await loader.load()

const splitter = new RecursiveCharacterTextSplitter({
    chunkOverlap:150,
    chunkSize:1000
})
const chunk = await splitter.splitDocuments(documents)

console.log(`Loaded ${chunk.length} chunk`);

//generation of model 
const extractor = await pipeline(
    "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"

)

const records = []


for (const [index,chunk1] of chunk.entries()){
 const result = await extractor(chunk1.pageContent,{
    pooling:"mean",
    normalize:true
 })   


records.push({
    documentId: DOCUMENT_ID,
    chunkIndex: index,
    text: chunk1.pageContent,
    metadata: chunk1.metadata,
    embedding: result.tolist()[0],
})

  console.log(`Processed ${index + 1}/${chunk.length}`);  console.log(`Processed ${index + 1}/${chunk.length}`);
}


//store  everything in mojnogodbv
try{
    await client.connect();
    const db = client.db("coaching_rag");
    const collection = db.collection("document_chunks")

    // Replace this document's previous ingestion
  await collection.deleteMany({ documentId: DOCUMENT_ID });

  if(records.length > 0){
        await collection.insertMany(records);
  }

  console.log(`Stored ${records.length} chunks in MongoDB`);

  const count = await collection.countDocuments({
    documentId: DOCUMENT_ID,
  });

  console.log("Verified stored chunks:", count);

}catch (error) {
  console.error("Storage failed:", error);
} finally {
  await client.close();
}