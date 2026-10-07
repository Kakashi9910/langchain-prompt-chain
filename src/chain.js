import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough } from "@langchain/core/runnables";

const llm = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: process.env.GEMINI_MODEL || "gemini-flash-lite-latest",
  temperature: 0.3,
  maxRetries: 5, // retries automatically on 429/503 overloads
});

const parser = new StringOutputParser();

// Step 1: topic -> outline
const outlineChain = PromptTemplate.fromTemplate(
  `You are a technical writer. Create a concise 4-point outline for an article
about "{topic}" aimed at {audience}. Return only the outline.`
)
  .pipe(llm)
  .pipe(parser);

// Step 2: outline -> draft
const draftChain = PromptTemplate.fromTemplate(
  `Write a short article (about 250 words) for {audience} that follows this outline.

OUTLINE:
{outline}`
)
  .pipe(llm)
  .pipe(parser);

// Step 3: draft -> polished final
const polishChain = PromptTemplate.fromTemplate(
  `Improve the clarity and flow of this article. Fix grammar, keep it under 250 words,
and add a one-line title at the top.

ARTICLE:
{draft}`
)
  .pipe(llm)
  .pipe(parser);

// Chain them: each step adds its output to the shared object, so later
// prompts can use {topic}, {audience}, {outline}, {draft}.
export const chain = RunnablePassthrough.assign({ outline: outlineChain })
  .pipe(RunnablePassthrough.assign({ draft: draftChain }))
  .pipe(RunnablePassthrough.assign({ final: polishChain }));

const result = await chain.invoke({
  topic: "how RAG chatbots work",
  audience: "beginner JavaScript developers",
});

console.log("=== OUTLINE ===\n" + result.outline);
console.log("\n=== DRAFT ===\n" + result.draft);
console.log("\n=== FINAL ===\n" + result.final);
