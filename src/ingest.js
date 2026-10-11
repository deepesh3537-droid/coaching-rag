// data  ingestion stage i.e. identiify data source prepare data and store daata 
import 'dotenv/config';

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const filePath = "../documents/fees.pdf";

//loading the pdf
const loader = new PDFLoader(filePath);
const  document = await loader.load();

console.log("Pages loaded: ",document)


const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:150
})

const chunks = await splitter.splitDocuments(document);

console.log("Chunks created:", chunks.length);

// 3. Inspect the output
console.log("\nFirst chunk:");
console.log(chunks[0]?.pageContent);

console.log("\nFirst chunk metadata:");
console.log(chunks[0]?.metadata);