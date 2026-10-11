import { PDFLoader

 } from "@langchain/community/document_loaders/fs/pdf";

 import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
 import { pipeline } from "@huggingface/transformers";

 const loader = new PDFLoader("../documents/fees.pdf");
 const document = await loader.load();

 //split the file into chunk

 const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:150
 })

 const chunks = await splitter.splitDocuments(document)

 console.log(`Chunks loaded: ${chunks.length}`);

 //loadin the model 

 const extractor = await pipeline(
    "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"

 )

 const result = await extractor(chunks[0].pageContent,{
    pooling:"mean",
    normalize: true
 })

const embedding = result.tolist();


console.log("Embedding dimensions:", embedding.length);
console.log("First 10 values:", embedding.slice(0, 10));
console.log("Original chunk:", chunks[0].pageContent);