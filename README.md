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

### 👤 User Prompt Example
> I'm feeling tired and unmotivated today.

### 🧠 Prompt Design Using RFTC

- **Role**: Motivational coach
- **Format**: JSON with 4 keys
- **Tone**: Encouraging and empathetic
- **Context**: User’s self-reported mood

---

## 🧠 Zero-Shot Prompting – MotivateMe

Provide a motivational quote that matches the user's current mood.  
Respond in JSON format with the keys: `mood`, `quote`, `author`, and `suggested_action`.  
The tone should be encouraging and supportive.

### ❓ Why is this Zero-Shot?

No examples are given to the model.  
It is expected to understand and complete the task based solely on the instruction.  
This tests the generalization power of the LLM.

---

## 📄 One-Shot Prompt Used

**Example Input:**  
"I’m feeling anxious today."

**Example Output:**
```json
{
  "mood": "anxious",
  "quote": "You don’t have to control your thoughts. You just have to stop letting them control you.",
  "author": "Dan Millman",
  "suggested_action": "Try a 2-minute breathing exercise."
}

---

## 📄 Multi-Shot Prompt

This prompt includes 3 input/output examples showing how to respond to different emotional states.

### Example 1:
Input: "I’m feeling anxious today."  
Output: *(structured JSON with mood-specific quote)*

### Example 2:
Input: "I feel tired and overwhelmed."  
Output: *(structured JSON with quote and action)*

### Example 3:
Input: "I'm feeling excited and motivated!"  
Output: *(structured JSON with a quote and creative suggestion)*

---

### New User Input:
"I’m feeling lost and confused."

The model should now follow the format, tone, and structure from the above examples and generate a similar motivational response.

---

## ❓ Why is this Multi-Shot?

The model is given **multiple examples** of the expected behavior before being asked to respond. This helps it learn tone, structure, and response logic more accurately.

---

## 📄 Dynamic Prompt Example

Suppose the user types:
> "I'm feeling extremely anxious and panicked."

The prompt generated **at runtime** would be:

> You are a motivational coach helping someone who is **extremely anxious and panicked**.  
> Provide an emotionally appropriate motivational quote.  
> Respond in JSON with the keys: `mood`, `quote`, `author`, and `suggested_action`.  
> Keep the tone calming and reassuring.

---

If the user input was:
> "I'm just a bit tired today."

The dynamically generated prompt would be:

> You are a motivational coach helping someone who is **just a bit tired**.  
> Provide a gentle motivational quote.  
> Respond in JSON... [etc.]

---

## ❓ Why is this Dynamic Prompting?

The prompt is **not static** — it is **constructed on the fly** based on the user's input and emotion intensity. This allows for highly personalized, context-aware responses.

---

# 🧠 Chain of Thought Prompting – Daily Motivation Bot

## 📄 Prompt Design

The user will give you a sentence describing their mood.  
First, identify the emotion(s) in the sentence.  
Then, reason about what type of motivational quote might help them.  
Finally, generate a motivational quote with a suitable tone.  
Respond in JSON format with keys: `mood`, `quote`, `author`, and `suggested_action`.  
Think step by step before giving the final output.

---

## 👤 User Input

"I’m feeling completely lost and unsure of what to do next."

---

## 🧠 Why is this Chain of Thought Prompting?

This prompt **forces the model to think step-by-step**.  
First, it breaks down the emotion, reasons about what quote is needed, and only then generates the final structured output.

This helps improve both **emotional accuracy** and **quote relevance**.

---
