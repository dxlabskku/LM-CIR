###################################
###### GENERAL PROMPTS ############
###################################

COT_PROMPT_TITLE_ONLY_GENERAL = '''
Your task is to update an image caption based on a reference image and a modification instruction.

You must follow a structured reasoning process and produce the output in JSON format.

---
## Instructions

### Step 1: Understand the reference image
- Identify the main objects and their relationships.
- Describe the image as completely as possible, including all visible objects and relevant attributes.

### Step 2: Understand the modification instruction
- Determine which object(s) or part of the image should be modified.
- Identify which attribute is being changed (e.g., type, position, color, presence, focus).

### Step 3: Imagine the modification
- Describe how the target object(s) change after the modification.
- Do NOT introduce new attributes unless explicitly required.
- If unspecified (e.g., “an animal”), keep it generic.

### Step 4: Generate the final caption
- Write ONE concise sentence describing the edited image.
- Focus ONLY on the modified object and changed attribute.
- Do NOT include irrelevant details from the original image.
- Do NOT describe objects that are not present in the final image.
- Do NOT use expressions like "with no" or "without".

---
## Important Rules
- If the instruction involves:
  - **Negation** (e.g., “not green”): do NOT describe that attribute.
  - **Removal** (e.g., “no people”): omit it, do NOT mention absence.
  - **Addition**: include only the added object + minimal context.
- Avoid hallucination (no guessing unspecified details).
- Keep the caption as short as possible.

---
## Input Format 

{
    "Original Image": <image_url>,
    "Manipulation text": <manipulation_text>.
}

---
## Output Format

In the actual response, return a JSON object with EXACTLY one field:

{
  "Target Image Description": "<final caption>"
}

---
## Internal Reasoning Format (for internal use only; do not output)

1. Understand the reference image:
...
2. Understand the modification instruction:
...
3. Imagine the modification:
...
4. Final Caption:
...

---
## Example 1
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Human and one animal from a different species from the last two."
}

Internal reasoning:
1. Understand the reference image:
The image shows two manta rays swimming underwater, surrounded by several small fish.
2. Understand the modification instruction:
The instruction requires replacing the current main subjects (the two manta rays) with a human and one animal of a different species.
3. Imagine the modification:
The manta rays are replaced by a human swimmer and a different type of animal. Since the instruction does not specify the exact animal species, the result should describe it only as an animal rather than assuming a specific one.
4. Final Caption:
A human and an animal swimming underwater.

<Response>
{
    "Target Image Description": "A human and an animal swimming underwater."
}

## Example 2
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "remove the lemon"
}

Internal reasoning:
1. Understand the reference image:
The image shows several pieces of cooked meat served on a plate, topped with slices of lemon and garnished with herbs.
2. Understand the modification instruction:
The instruction requires removing the lemon slices while keeping the rest unchanged.
3. Imagine the modification:
The lemon slices are removed, leaving the cooked meat and herbs. The caption should not mention the removed lemon or include unrelated details.
4. Final Caption:
Cooked meat.

<Response>
{
    "Target Image Description": "Cooked meat."
}

## Example 3
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "more focused on its head"
}

Internal reasoning:
1. Understand the reference image:
The image shows a small puppy lying on the ground, resting with its eyes closed.
2. Understand the modification instruction:
The instruction requires shifting the focus to the puppy’s head.
3. Imagine the modification:
The image emphasizes the puppy’s head while reducing attention to other parts. The caption should reflect this change in focus without introducing new or stylistic details.
4. Final Caption:
A puppy’s head.

<Response>
{
    "Target Image Description": "A puppy’s head."
}

## Example 4
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "White sits in front of white mess."
}

Internal reasoning:
1. Understand the reference image:
The image shows three puppies.
2. Understand the modification instruction:
The instruction describes a white subject sitting in front of a white mess.
3. Imagine the modification:
The white subject is grounded as a dog from the image, while "white mess" remains as stated since it is not further specified. The caption should follow the instruction without adding extra details.
4. Final Caption:
A white dog sitting in front of a white mess.

<Response>
{
    "Target Image Description": "A white dog sitting in front of a white mess."
}

## Example 5
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Put a chair in the middle."
}

Internal reasoning:
1. Understand the reference image:
The image shows a room with large windows, a desk, and some furniture.
2. Understand the modification instruction:
The instruction requires adding a chair in the middle of the scene.
3. Imagine the modification:
A chair is placed in the middle of the room. The caption should reflect this addition without introducing extra details.
4. Final Caption:
A chair in the middle of a room.

<Response>
{
    "Target Image Description": "A chair in the middle of a room."
}
'''


COT_PROMPT_TITLE_ONLY_FASHION_IQ = '''
Your task is to update an image caption based on a reference image and a modification instruction.

You must follow a structured reasoning process and produce the output in JSON format.

---
## Instructions

### Step 1: Understand the reference image
- Identify the main objects and their relationships.
- Describe the image as completely as possible, including all visible objects and relevant attributes.

### Step 2: Understand the modification instruction
- Determine which object(s) or part of the image should be modified.
- Identify which attribute is being changed (e.g., type, position, color, presence, focus).

### Step 3: Imagine the modification
- Describe how the target object(s) change after the modification.
- Do NOT introduce new attributes unless explicitly required.
- If unspecified (e.g., “an animal”), keep it generic.

### Step 4: Generate the final caption
- Write ONE concise sentence describing the edited image.
- Focus ONLY on the modified object and changed attribute.
- Do NOT include irrelevant details from the original image.
- Do NOT describe objects that are not present in the final image.
- Do NOT use expressions like "with no" or "without".

---
## Important Rules
- If the instruction involves:
  - **Negation** (e.g., “not green”): do NOT describe that attribute.
  - **Removal** (e.g., “no people”): omit it, do NOT mention absence.
  - **Addition**: include only the added object + minimal context.
- Avoid hallucination (no guessing unspecified details).
- Keep the caption as short as possible.

---
## Input Format 

{
    "Original Image": <image_url>,
    "Manipulation text": <manipulation_text>.
}

---
## Output Format

In the actual response, return a JSON object with EXACTLY one field:

{
  "Target Image Description": "<final caption>"
}

---
## Internal Reasoning Format (for internal use only; do not output)

1. Understand the reference image:
...
2. Understand the modification instruction:
...
3. Imagine the modification:
...
4. Final Caption:
...

---
## Example 1
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "is longer and more elegant. is longer and sleeveless"
}

Internal reasoning:
1. Understand the reference image:
A short dress with one sleeve.
2. Understand the modification instruction:
The instruction requires changing the dress to be longer, sleeveless, and more elegant.
3. Imagine the modification:
The dress is modified to be longer and without sleeves, with a more elegant style. The caption should describe only the garment and its attributes, without mentioning any person or actions.
4. Final Caption:
A long, sleeveless, elegant dress.

<Response>
{
    "Target Image Description": "A long, sleeveless, elegant dress."
}

## Example 2
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "has longer sleeves and a different color. has longer sleeves and is more casual"
}

Internal reasoning:
1. Understand the reference image:
A woman wearing a pink sleeveless dress.
2. Understand the modification instruction:
The instruction requires changing the dress to have longer sleeves, a different color, and a more casual style.
3. Imagine the modification:
The dress is modified with long sleeves and a more casual appearance. Since the new color is unspecified, it is not described. The caption should describe only the garment and its attributes, without mentioning any person or actions.
4. Final Caption:
A casual long-sleeved dress.

<Response>
{
    "Target Image Description": "A casual long-sleeved dress."
}

## Example 3
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "is darker and has shorter sleeves. does not cover the sholders"
}

Internal reasoning:
1. Understand the reference image:
A woman wearing a patterned dress with short sleeves.
2. Understand the modification instruction:
The instruction requires making the dress darker, shortening the sleeves, and removing shoulder coverage.
3. Imagine the modification:
The dress is modified to have a darker appearance and no sleeves. The phrase "does not cover the shoulders" is interpreted as equivalent to being sleeveless. The caption should describe only the garment and its attributes, without mentioning any person or actions.
4. Final Caption:
A dark sleeveless dress.

<Response>
{
    "Target Image Description": "A dark sleeveless dress."
}

## Example 4
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "is white and less revealing. is white and same length all around bottom"
}

Internal reasoning:
1. Understand the reference image:
A woman wearing a black sleeveless dress with an uneven hemline that is above the knee.
2. Understand the modification instruction:
The instruction requires changing the dress to be white, more modest, and to have a uniform hemline while remaining above the knee.
3. Imagine the modification:
The dress is modified to be white with a more modest style. The hemline is adjusted to be even all around, and it remains above the knee. The caption should describe only the garment and its attributes, without mentioning any person or actions.
4. Final Caption:
A modest white dress with an even above-the-knee hemline.

<Response>
{
    "Target Image Description": "A modest white dress with an even above-the-knee hemline."
}

## Example 5
<Input>
{
    "Original Image": <image_url>,
    "Edited Caption": "A gray dress that is shorter and features a ruffled hem."
}

Internal reasoning:
1. Understand the reference image:
A woman wearing a long strapless dress.
2. Understand the modification instruction:
The instruction requires changing the dress to be gray, shorter in length, and to have ruffled details.
3. Imagine the modification:
The dress is modified to be gray with a shorter length and added ruffled details. The caption should describe only the garment and its attributes, without mentioning any person or actions.
4. Final Caption:
A short gray dress with a ruffled hem.

<Response>
{
    "Target Image Description": "A short gray dress with a ruffled hem."
}
'''

CRITIC_PROMPT = """\
You are evaluating a candidate image retrieved for a Composed Image Retrieval task.

The user provides a REFERENCE image and asks for an image that applies a specific MODIFICATION to it. We have a CANDIDATE image and want to judge how good a match it is.

Modification request: "{mod_text}"

Rate the CANDIDATE on TWO 0-10 integer scales:

1. EDIT  -- How well does the CANDIDATE reflect the modification described above,
            relative to the REFERENCE image?
            (0 = the modification is not applied at all,
             10 = the modification is applied exactly as requested)

2. PRESERVE -- How well does the CANDIDATE preserve the non-modified attributes
               of the REFERENCE (subject identity, scene/category, layout, style,
               anything not mentioned in the modification)?
               (0 = unrelated to the reference,
                10 = same subject/scene with only the requested change)

Be strict. Use the full 0-10 range. If the candidate is wrong, give low scores.

Return ONLY a JSON object on a single line, no extra text, no markdown fences:
{{"edit": <int 0-10>, "preserve": <int 0-10>, "reason": "<one short sentence>"}}
"""
