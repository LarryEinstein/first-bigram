# bigram_model.py
import torch
import numpy as np

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"length of dataset in characters: {len(text)}")

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"chars are: ",''.join(chars))
print(f"vocab size is: ", vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# Now let's build the bigram count matrix
# This will be a vocab_size x vocab_size matrix where entry [i,j] represents
# how many times character i is followed by character j

# Initialize the count matrix with zeros
encoded = torch.tensor(encode(text), dtype=torch.long)
idx = encoded[:-1] * vocab_size + encoded[1:]
bigram_counts = torch.bincount(idx, minlength=vocab_size**2)\
                    .reshape(vocab_size, vocab_size)

print(f"Bigram count matrix shape: {bigram_counts.shape}")
print(f"Total bigrams counted: {bigram_counts.sum().item()}")

# Let's look at some examples
print(f"\nSome example bigram counts:")
for i in range(min(5, vocab_size)):
    for j in range(min(5, vocab_size)):
        if bigram_counts[i, j] > 0:
            char_i = itos[i]
            char_j = itos[j]
            count = bigram_counts[i, j].item()
            print(f"'{char_i}' -> '{char_j}': {count} times")

# Let's also check a specific example you mentioned
# Find the most common bigrams
max_count = bigram_counts.max().item()
max_positions = (bigram_counts == max_count).nonzero()
if len(max_positions) > 0:
    i, j = max_positions[0]
    char_i = itos[i.item()]
    char_j = itos[j.item()]
    print(f"\nMost common bigram: '{char_i}' -> '{char_j}' appears {max_count} times")

print(f"\nBigram count matrix created successfully!")
print(f"Matrix dimensions: {vocab_size} x {vocab_size}")

# Convert counts to probabilities for text generation
print(f"\nConverting counts to probabilities...")
bigram_probs = bigram_counts.float()

# Normalize each row to get probabilities (add small epsilon to avoid division by zero)
row_sums = bigram_probs.sum(dim=1, keepdim=True)
# Add small epsilon only where row sum is 0 to avoid changing existing probabilities
epsilon = 1e-8
row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums) * epsilon, row_sums)
bigram_probs = bigram_probs / row_sums

print(f"Probability matrix created!")

# Text generation function
def generate_next_char(current_char):
    """Generate the next character based on bigram probabilities"""
    if current_char not in stoi:
        print(f"Character '{current_char}' not in vocabulary!")
        return None
    
    current_idx = stoi[current_char]
    probs = bigram_probs[current_idx]
    
    # Sample from the probability distribution
    next_idx = torch.multinomial(probs, 1).item()
    next_char = itos[next_idx]
    
    return next_char

def show_probabilities(current_char, top_n=10):
    """Show the top N most likely next characters and their probabilities"""
    if current_char not in stoi:
        print(f"Character '{current_char}' not in vocabulary!")
        return
    
    current_idx = stoi[current_char]
    probs = bigram_probs[current_idx]
    
    # Get top probabilities
    top_probs, top_indices = torch.topk(probs, min(top_n, vocab_size))
    
    print(f"\nTop {min(top_n, vocab_size)} most likely characters after '{current_char}':")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        char = itos[idx.item()]
        prob_val = prob.item()
        if prob_val > 0:
            char_display = char if char not in ['\n', '\t', ' '] else ('\\n' if char == '\n' else ('\\t' if char == '\t' else 'SPACE'))
            print(f"  {i+1}. '{char_display}': {prob_val:.4f} ({prob_val*100:.2f}%)")

# Interactive text generation
print(f"\n" + "="*50)
print(f"BIGRAM TEXT GENERATION")
print(f"="*50)

def interactive_generation():
    """Interactive text generation loop"""
    print(f"\nStarting interactive text generation!")
    print(f"Commands:")
    print(f"  - Type a single character to start generation")
    print(f"  - Press Enter to generate the next character")
    print(f"  - Type 'show' to see probabilities for current character")
    print(f"  - Type 'reset' to start over")
    print(f"  - Type 'quit' to exit")
    
    current_char = None
    generated_text = ""
    
    while True:
        if current_char is None:
            user_input = input(f"\nEnter starting character: ").strip()
            if user_input.lower() == 'quit':
                break
            elif len(user_input) == 1 and user_input in stoi:
                current_char = user_input
                generated_text = current_char
                print(f"Starting with: '{current_char}'")
                print(f"Generated text so far: {generated_text}")
                show_probabilities(current_char, 5)
            else:
                print(f"Please enter a single character from the vocabulary!")
                print(f"Available characters: {''.join(chars)}")
        else:
            user_input = input(f"\nPress Enter to generate next character (or 'show'/'reset'/'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'reset':
                current_char = None
                generated_text = ""
                print(f"Reset! Starting over...")
                continue
            elif user_input.lower() == 'show':
                show_probabilities(current_char, 10)
                continue
            elif user_input == "":
                # Generate next character
                next_char = generate_next_char(current_char)
                if next_char:
                    generated_text += next_char
                    current_char = next_char
                    
                    # Display the character nicely
                    display_char = next_char if next_char not in ['\n', '\t'] else ('\\n' if next_char == '\n' else '\\t')
                    print(f"Generated: '{display_char}'")
                    print(f"Generated text so far: {repr(generated_text)}")
                    show_probabilities(current_char, 5)
            else:
                print(f"Unknown command. Use Enter, 'show', 'reset', or 'quit'")

# Run interactive generation
interactive_generation()

# Create HTML visualization
print(f"\nCreating HTML visualization...")

def char_display(char):
    """Convert special characters to readable format"""
    if char == '\n':
        return '\\n'
    elif char == '\t':
        return '\\t'
    elif char == ' ':
        return 'SPACE'
    elif char == '"':
        return '&quot;'
    elif char == '<':
        return '&lt;'
    elif char == '>':
        return '&gt;'
    elif char == '&':
        return '&amp;'
    else:
        return char

# Convert to numpy for easier processing
bigram_numpy = bigram_counts.numpy()

# Create top bigrams list
bigram_list = []
for i in range(vocab_size):
    for j in range(vocab_size):
        count = bigram_counts[i, j].item()
        if count > 0:
            bigram_list.append({
                'first_char': chars[i],
                'second_char': chars[j],
                'count': count,
                'first_idx': i,
                'second_idx': j
            })

# Sort by count
bigram_list.sort(key=lambda x: x['count'], reverse=True)

# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bigram Language Model Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
            text-align: center;
        }}
        .stats {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .heatmap-container {{
            overflow: auto;
            max-height: 600px;
            border: 1px solid #ddd;
            margin: 20px 0;
        }}
        .heatmap {{
            border-collapse: collapse;
            font-size: 10px;
        }}
        .heatmap th, .heatmap td {{
            width: 20px;
            height: 20px;
            text-align: center;
            border: 1px solid #eee;
            padding: 2px;
        }}
        .heatmap th {{
            background-color: #f0f0f0;
            font-weight: bold;
        }}
        .top-bigrams {{
            margin: 20px 0;
        }}
        .bigram-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        .bigram-table th, .bigram-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .bigram-table th {{
            background-color: #f2f2f2;
        }}
        .bigram-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .char-display {{
            font-family: monospace;
            background-color: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bigram Language Model Visualization</h1>
        <h2>Shakespeare Dataset Analysis</h2>
        
        <div class="stats">
            <h3>Dataset Statistics</h3>
            <p><strong>Total characters:</strong> {len(text):,}</p>
            <p><strong>Vocabulary size:</strong> {vocab_size}</p>
            <p><strong>Total bigrams:</strong> {bigram_counts.sum().item():,}</p>
            <p><strong>Unique bigrams:</strong> {len(bigram_list):,}</p>
        </div>

        <div class="top-bigrams">
            <h3>Top 20 Most Common Bigrams</h3>
            <table class="bigram-table">
                <tr>
                    <th>Rank</th>
                    <th>First Character</th>
                    <th>Second Character</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
"""

# Add top 20 bigrams to HTML
total_bigrams = bigram_counts.sum().item()
for idx, bigram in enumerate(bigram_list[:20]):
    percentage = (bigram['count'] / total_bigrams) * 100
    html_content += f"""
                <tr>
                    <td>{idx + 1}</td>
                    <td><span class="char-display">{char_display(bigram['first_char'])}</span></td>
                    <td><span class="char-display">{char_display(bigram['second_char'])}</span></td>
                    <td>{bigram['count']:,}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
"""

html_content += """
            </table>
        </div>

        <div class="heatmap-container">
            <h3>Bigram Count Heatmap</h3>
            <p><em>Darker colors indicate higher counts. Hover over cells to see details.</em></p>
            <table class="heatmap">
                <tr>
                    <th></th>
"""

# Add column headers
for char in chars:
    html_content += f'<th title="{char_display(char)}">{char_display(char)}</th>'

html_content += "</tr>"

# Add heatmap rows
max_count = bigram_counts.max().item()
for i, char_i in enumerate(chars):
    html_content += f'<tr><th title="{char_display(char_i)}">{char_display(char_i)}</th>'
    for j, char_j in enumerate(chars):
        count = bigram_counts[i, j].item()
        if count > 0:
            # Color intensity based on count (logarithmic scale for better visualization)
            intensity = min(255, int(255 * (np.log(count + 1) / np.log(max_count + 1))))
            color = f"rgb({255-intensity}, {255-intensity}, 255)"
            html_content += f'<td style="background-color: {color}" title="{char_display(char_i)} → {char_display(char_j)}: {count}">{count if count < 1000 else "999+"}</td>'
        else:
            html_content += '<td style="background-color: white" title="0">0</td>'
    html_content += "</tr>"

html_content += """
            </table>
        </div>
        
        <div class="stats">
            <h3>Character Frequency Analysis</h3>
            <p>Most common starting characters (first in bigrams):</p>
            <ul>
"""

# Calculate character frequencies as first character
first_char_counts = bigram_counts.sum(dim=1)
first_char_sorted = torch.argsort(first_char_counts, descending=True)

for i in range(min(10, len(first_char_sorted))):
    char_idx = first_char_sorted[i].item()
    char = chars[char_idx]
    count = first_char_counts[char_idx].item()
    html_content += f'<li><span class="char-display">{char_display(char)}</span>: {count:,} times</li>'

html_content += """
            </ul>
        </div>
    </div>
</body>
</html>
"""

# Save HTML file
with open('bigram_visualization.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML visualization saved to 'bigram_visualization.html'")
print(f"Open this file in your web browser to view the interactive visualization!")
print(f"The visualization includes:")
print(f"  - Dataset statistics")
print(f"  - Top 20 most common bigrams")
print(f"  - Interactive heatmap of the full bigram matrix")
print(f"  - Character frequency analysis") 