# convert_images_to_descriptions.py

import openai
import os
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

def image_to_base64(image_path):
    img = Image.open(image_path).convert("RGB")  # Handles .webp, .jpg, .png
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_description(image_path):
    img_b64 = image_to_base64(image_path)

    prompt = """You are a helpful virtual home assistant guiding a house tour. Describe the room shown in this image.
Focus on the layout, size, window placement, natural lighting, flow between spaces, and the overall ambiance.
Avoid describing furniture—emphasize architectural features, space utilization, and how the room might feel to walk through. Keep the tone factual, friendly, and tour-style."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"  # Important: jpeg here
                        }
                    }
                ]
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # Set your local folder
    main_folder = "/Users/truptaditya/Documents/GitHub/HouseTour/Images"  # Example: "C:/Users/yourname/Documents/HouseTour/Images"
    
    # 1. Save Property Details to Text File
    property_details_content = """
Property Details :
Basic Property Details
Address: 2848 Chromite Dr, Santa Clara, CA 95051

Property Type: Single Family Residence

Year Built : 1958

Lot Size : 5600 square feet lot

Home Size (Sqft) : 1383 Sqft

Number of Bedrooms / Bathrooms : 4 beds 2 bath

Stories (Single / Multi-Level)  : Single level

Parking Info (Garage, Driveway, etc.)

HOA Fee (if applicable)

What's special
New electrical panel, Gourmet kitchen, Waterproof wood flooring, Custom wood cabinetry, Quartz countertops, Stainless steel appliances, Led recessed lights
Welcome to this charming, beautifully remodeled single-family home in the heart of Santa Clara, just 1 mile from NVIDIA. Offering 4 bedrooms, 2 bathrooms, 1,383 SqFt of living space plus a 236 SqFt bonus room on a 5,600 SqFt lot. Features include a new electrical panel, HVAC system, and plumbing. The gourmet kitchen boasts custom wood cabinetry, quartz countertops, and stainless steel appliances. Bathrooms have new vanities, faucets, mirrors, toilets, and doors. High-end flooring, fresh interior/exterior paint, new double-paned windows, and 30 LED recessed lights enhance the home. Nearby highlights: Central Park: A picturesque park with walking trails, playgrounds, picnic areas, and a lake. California's Great America: A thrilling amusement park with rides, shows, and attractions for all ages. Levi's Stadium: Home of the San Francisco 49ers and host to various sporting events and concerts. Close proximity to Major Tech Giants like NVIDIA, Apple, Microsoft, Google, LinkedIn, and Facebook/Meta. Convenient Transportation: Easy access to public transit & highways. Top-rated Schools: Excellent options within 2 miles. Experience modern living in a prime location, combining convenience, education, and entertainment!

Facts & features
Interior

Bedrooms & bathrooms
Bedrooms: 4
Bathrooms: 2
Full bathrooms: 2
Rooms
Room types: Bonus Room
Bathroom
Features: Shower and Tub, Skylight, Tile, Updated Baths
Dining room
Features: Dining Family Combo, Skylights
Family room
Features: Separate Family Room
Kitchen
Features: Exhaust Fan
Heating
Central Forced Air
Cooling
Central Air
Appliances
Included: Dishwasher, Exhaust Fan, Freezer, Disposal, Ice Maker, Gas Oven/Range, Refrigerator
Features
One Or More Skylights
Flooring: Wood
Number of fireplaces: 1
Fireplace features: Family Room
Interior area
Total structure area: 1,383
Total interior livable area: 1,383 sqft

Property

Parking
Total spaces: 2
Parking features: Attached
Attached garage spaces: 2
Features
Stories: 1
Patio & porch: Balcony/Patio
Exterior features: Back Yard, Barbecue, Fenced, Gazebo
Lot
Size: 5,600 Square Feet
Details
Parcel number: 21617078
Zoning: R1
Special conditions: Standard

Construction

Type & style
Home type: Single Family
Property subtype: Single Family Residence,
Materials
Foundation: Crawl Space
Roof: Shingle
Condition
New construction: No
Year built: 1958

Utilities & Green Energy
Gas: Public Utilities
Sewer: Public Sewer
Water: Public
Utilities for property: Public Utilities, Water Public

Community & HOA
Location
Region: Santa Clara

Financial & listing details
Price per square foot: $1,365/sqft
Tax assessed value: $87,366
Annual tax amount: $1,160
Date on market: 3/19/2025
Listing agreement: Exclusive Right To Sell
Listing terms: Cash or Conventional Loan
"""
    property_details_path = os.path.join(main_folder, "property_details.txt")
    
    # Save property details
    with open(property_details_path, "w", encoding="utf-8") as file:
        file.write(property_details_content)
    
    print(f"✅ Property details saved at: {property_details_path}")

    # 2. Generate Descriptions for Rooms
    output_file = os.path.join(main_folder, "room_description.txt")
    print(f"Creating/Opening output file at: {output_file}")

    if not os.path.exists(main_folder):
        raise Exception(f"Main folder does not exist: {main_folder}")

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for room_folder in os.listdir(main_folder):
            room_path = os.path.join(main_folder, room_folder)
            if os.path.isdir(room_path):
                print(f"\n📂 Processing: {room_folder}")
                out_file.write(f"=== {room_folder} ===\n\n")

                for filename in sorted(os.listdir(room_path)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_path = os.path.join(room_path, filename)
                        try:
                            print(f"  - Describing: {filename}")
                            desc = generate_description(image_path)
                            print(f"  - Description: {desc}")
                            out_file.write(f"{filename}:\n{desc}\n\n")
                        except Exception as e:
                            print(f"  ❌ Error with {filename}: {e}")

    # 3. Also create a description.txt inside each room folder
    for room_folder in os.listdir(main_folder):
        room_path = os.path.join(main_folder, room_folder)
        if os.path.isdir(room_path):
            room_output_file = os.path.join(room_path, "room_description.txt")
            print(f"\n📂 Processing for individual folder: {room_folder}")
            with open(room_output_file, 'w', encoding='utf-8') as f:
                for filename in sorted(os.listdir(room_path)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_path = os.path.join(room_path, filename)
                        try:
                            print(f"  - Describing: {filename}")
                            desc = generate_description(image_path)
                            print(f"  - Description: {desc}")
                            f.write(f"{filename}:\n{desc}\n\n")
                        except Exception as e:
                            print(f"  ❌ Error with {filename}: {e}")
    print(f"\n✅ Room descriptions saved at: {output_file}")
    print(f"✅ Individual room descriptions saved in each folder.") 