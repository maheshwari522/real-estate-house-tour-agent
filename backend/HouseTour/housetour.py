# house_tour.py
from openai import OpenAI
import os
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# from IPython.display import Audio  # Comment or remove if not using Jupyter

# Create OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or "your-openai-api-key-here"
)
# ✅ Paths
property_details_path = "/Users/truptaditya/Documents/GitHub/HouseTour/Images/property_details.txt"
room_descriptions_path = "/Users/truptaditya/Documents/GitHub/HouseTour/Images/room_description.txt"
output_script_path = "/Users/truptaditya/Documents/GitHub/HouseTour/Images/house_tour_script.txt"
output_audio_path = "/Users/truptaditya/Documents/GitHub/HouseTour/Images/house_tour_audio.mp3"

# ✅ Create House Tour Script
def create_house_tour_script(property_info, room_descriptions):
    prompt = f"""
You are a virtual tour guide for a modern smart home. Your task is to take the property overview and room descriptions and generate a natural, immersive house tour — just like a museum guide would.

Property Details:
{property_info}

Room Descriptions:
{room_descriptions}

Now, generate a single, continuous house tour script:
- Start with a warm welcome and a brief overview of the home using the property details.
- Walk the listener from the entrance through each room in a logical order.
- Use directional cues (e.g., "To your right is...", "As you walk ahead...")
- Make it conversational, friendly, and informative.
- Avoid repeating the same phrases.
- End with a thank-you and invite them to explore further.

Output only the final script.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.75
    )

    return response.choices[0].message.content.strip()

# ✅ Main function
def main():
    # Load property and room data
    with open(property_details_path, "r", encoding="utf-8") as f:
        property_info = f.read()

    with open(room_descriptions_path, "r", encoding="utf-8") as f:
        room_descriptions = f.read()

    # Generate tour script
    tour_script = create_house_tour_script(property_info, room_descriptions)

    # Save tour script
    with open(output_script_path, "w", encoding="utf-8") as f:
        f.write(tour_script)

    print("\n✅ House Tour Script Generated!")
    print(tour_script)

    # Convert to audio
    tts = gTTS(text=tour_script, lang='en')
    tts.save(output_audio_path)

    print("\n✅ Audio file generated at:", output_audio_path)

    # Optional: Play audio if you want
    # Audio(output_audio_path)  # Only works inside Jupyter/Colab

if __name__ == "__main__":
    main()
