# 🧠 Daily Motivation Bot - GenAI Project

Welcome to the **Daily Motivation Bot**, a CLI-based GenAI agent that helps users stay motivated by generating personalized quotes based on their current mood.

This project was created as part of the **Kalvium course on Developing AI Agents using GenAI**.

---

## 💡 Project Overview

The Daily Motivation Bot asks the user how they feel and responds with a motivational quote tailored to their mood. It uses Generative AI techniques to reason, retrieve context, format output, and call functions to log or save results.

---

## 🚀 Features & Concepts Implemented

### ✅ 1. Prompting
The agent prompts the user:  
> "How are you feeling today?"  

It uses structured prompts (RFTC) to ensure tone, format, and context are appropriate for delivering supportive motivation.

### ✅ 2. Retrieval-Augmented Generation (RAG)
Instead of relying solely on model memory, the bot retrieves relevant quotes from a local or vector database based on the user's input mood (e.g., "tired", "anxious", "lost").

### ✅ 3. Structured Output
The agent formats its response as structured JSON:
```json
{
  "mood": "tired",
  "quote": "Rest if you must, but don’t you quit.",
  "author": "Unknown",
  "suggested_action": "Take a short walk and hydrate."
}
```

### ✅ 4. Function Calling
The agent uses function calling to trigger post-processing actions like:

Saving the quote to a local file

(Later) Emailing the quote to the user via external API

### 🧪 Tech Stack
Language: Python (CLI-based)

LLM Integration: `Google Studio API` for prompting and generation

Planned Additions: Email reminders, calendar integration, quote history

---

## 🤖 System Prompt
You are a friendly and empathetic motivational coach. Your job is to respond to a user's emotional state with a motivational quote that matches their mood. Always respond in JSON format with the following keys: `mood`, `quote`, `author`, and `suggested_action`. Keep your tone encouraging and supportive.

## 👤 User Prompt Example
I'm feeling tired and unmotivated today.

## 🧠 Prompt Design Using RFTC

- **Role**: Motivational coach
- **Format**: JSON with 4 keys
- **Tone**: Encouraging and empathetic
- **Context**: User’s self-reported mood
