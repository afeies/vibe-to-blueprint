import os

if os.environ.get("MOCK"):
    from pipeline.mock import parse_prompt, make_edge_map, generate_images, rank_images
else:
    from pipeline.parser import parse_prompt
    from pipeline.layout import make_edge_map
    from pipeline.generate import generate_images
    from pipeline.rank import rank_images

def run(user_text: str):
    print("\n--- Parsing prompt ---")
    schema = parse_prompt(user_text)
    print(schema)

    print("\n--- Generating edge map ---")
    edge_map = make_edge_map(schema.get("rooms", ["living room"]))
    edge_map.save("outputs/edge_map.png")

    print("\n--- Generating images ---")
    images = generate_images(schema, edge_map, n=1)

    print("\n--- Ranking images ---")
    ranked = rank_images(images, user_text)

    for i, img in enumerate(ranked):
        path = f"outputs/result_{i+1}.png"
        img.save(path)
        print(f"Saved {path}")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Create a list to store the conversation history
    history = []

    while True:
        txt = input("\nDescribe your room vibe (or type 'done' to quit): ")

        if txt.strip().lower() == "done":
            break

        # Add the new request to the history
        history.append(txt)

        # Combine all history into one big context string for the AI
        full_context = " | ".join(history)

        # Pass the combined history to your run function
        run(full_context)