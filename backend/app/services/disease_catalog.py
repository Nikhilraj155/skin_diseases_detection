# Comprehensive disease catalog with detailed information
# Used for generating PDF reports

DISEASE_CATALOG = {
    "Atopic_Dermatitis": {
        "name": "Atopic Dermatitis (Eczema)",
        "description": "Atopic dermatitis is a chronic inflammatory skin condition that often causes dry, itchy, and irritated skin. It's commonly seen in children but can affect people of all ages.",
        "symptoms": [
            "Dry, scaly skin",
            "Intense itching, especially at night",
            "Red to brownish-gray patches",
            "Small, raised bumps that may leak fluid",
            "Thickened, cracked skin",
            "Raw, sensitive skin from scratching"
        ],
        "risk_factors": [
            "Family history of eczema, allergies, or asthma",
            "Living in cold, dry climates",
            "Stress and emotional factors",
            "Exposure to irritants (soaps, detergents)"
        ],
        "recommendations": [
            "Moisturize skin regularly with fragrance-free creams",
            "Use mild, fragrance-free soaps and detergents",
            "Avoid hot showers; use lukewarm water",
            "Wear soft, breathable fabrics like cotton",
            "Keep nails short to prevent skin damage from scratching",
            "Consult a dermatologist for prescribed topical treatments",
            "Consider antihistamines for itching relief"
        ],
        "when_to_see_doctor": "If the rash is widespread, severe, or not responding to home care. Seek immediate care if the skin shows signs of infection (increased redness, warmth, swelling, or pus)."
    },
    "Basal_Cell_Carcinoma": {
        "name": "Basal Cell Carcinoma",
        "description": "Basal cell carcinoma is the most common type of skin cancer. It usually develops in sun-exposed areas and grows slowly, rarely spreading to other parts of the body.",
        "symptoms": [
            "Pearly or waxy bump on skin",
            "Flat, flesh-colored or brown scar-like lesion",
            "Pink or red patch that may be itchy",
            "Open sore that doesn't heal",
            "Visible blood vessels in the lesion",
            "Central depression or ulceration"
        ],
        "risk_factors": [
            "Chronic sun exposure",
            "History of sunburns",
            "Fair skin, light eyes, or blond/red hair",
            "Previous skin cancer",
            "Radiation therapy",
            "Immunosuppression"
        ],
        "recommendations": [
            "Schedule immediate appointment with dermatologist",
            "Avoid further sun exposure; use SPF 30+ sunscreen",
            "Wear protective clothing and hats",
            "Perform monthly self-skin examinations",
            "Photograph suspicious lesions for monitoring",
            "Treatment options include surgical removal, Mohs surgery, or topical medications"
        ],
        "when_to_see_doctor": "Immediately. Basal cell carcinoma requires professional evaluation and treatment. Early detection leads to better outcomes."
    },
    "Benign_Keratosis": {
        "name": "Benign Keratosis",
        "description": "Benign keratosis refers to non-cancerous skin growths that can appear as rough, thickened, or wart-like lesions. Includes seborrheic keratosis and other benign growths.",
        "symptoms": [
            "Waxy or stuck-on appearance",
            "Brown, black, or tan color",
            "Rough or scaly texture",
            "Varies in size from small to several inches",
            "Usually appears on face, chest, shoulders, or back",
            "May be itchy or irritated by clothing"
        ],
        "risk_factors": [
            "Age (more common after age 40)",
            "Family history",
            "Sun exposure",
            "Fair skin type"
        ],
        "recommendations": [
            "Generally no treatment needed if not bothersome",
            "Keep the area moisturized if dry",
            "Avoid picking or scratching the growths",
            "Use gentle skincare products",
            "Consult dermatologist if growth changes, bleeds, or causes discomfort",
            "Removal options include cryotherapy, curettage, or laser therapy if desired"
        ],
        "when_to_see_doctor": "If the growth changes in size, color, or shape, bleeds, or causes discomfort. Also consult if you have many growths to rule out other conditions."
    },
    "Dermatofibroma": {
        "name": "Dermatofibroma",
        "description": "Dermatofibroma is a usually harmless skin nodule that feels firm and commonly develops on the arms or legs. It's often called a benign fibrous histiocytoma.",
        "symptoms": [
            "Firm, raised bump under the skin",
            "Size usually less than 1 cm",
            "Pink, red, brown, or purple color",
            "May dimple or indent when pressed",
            "Usually on legs, arms, or torso",
            "May be itchy or tender"
        ],
        "risk_factors": [
            "Age (most common in young to middle-aged adults)",
            "Minor skin trauma or insect bites",
            "More common in women"
        ],
        "recommendations": [
            "No treatment needed if not bothersome",
            "Avoid scratching or irritating the area",
            "Apply moisturizer if dry",
            "Protect from sun exposure",
            "Monitor for any changes in size, color, or symptoms",
            "Removal only if causing discomfort or for cosmetic reasons"
        ],
        "when_to_see_doctor": "If the lesion changes significantly, becomes painful, or if you're unsure about the diagnosis."
    },
    "Eczema": {
        "name": "Eczema (Dermatitis)",
        "description": "Eczema is a broad term for skin inflammation that can cause itchiness, redness, dryness, and rash flare-ups. It includes various types like contact dermatitis and atopic dermatitis.",
        "symptoms": [
            "Red, inflamed skin",
            "Intense itching",
            "Dry, sensitive skin",
            "Rough, leathery patches",
            "Small bumps that may ooze",
            "Swelling"
        ],
        "risk_factors": [
            "Family history of eczema or allergies",
            "Asthma or hay fever",
            "Environmental triggers",
            "Stress",
            "Skin barrier defects"
        ],
        "recommendations": [
            "Identify and avoid triggers",
            "Moisturize skin at least twice daily",
            "Use mild, fragrance-free products",
            "Take short, lukewarm baths",
            "Wear soft, breathable fabrics",
            "Keep nails short to prevent scratching",
            "Use prescribed topical corticosteroids as directed",
            "Consider wet wrap therapy for severe flare-ups"
        ],
        "when_to_see_doctor": "If over-the-counter treatments aren't helping, if the rash is severe or widespread, or if signs of infection appear."
    },
    "Fungal_Infections": {
        "name": "Fungal Infections",
        "description": "Fungal skin infections are caused by fungi and may produce red, itchy, scaly, or ring-shaped skin changes. Common types include athlete's foot, ringworm, and yeast infections.",
        "symptoms": [
            "Red, itchy rash",
            "Scaling or flaking skin",
            "Ring-shaped lesions (ringworm)",
            "Blisters or pustules",
            "Peeling or cracking skin",
            "Musty or unpleasant odor"
        ],
        "risk_factors": [
            "Warm, moist environments",
            "Shared公共 areas (pools, gyms, locker rooms)",
            "Wet or sweaty skin",
            "Weakened immune system",
            "Diabetes",
            "Tight or non-breathable clothing"
        ],
        "recommendations": [
            "Keep affected area clean and dry",
            "Apply antifungal creams or powders",
            "Change socks and underwear daily",
            "Wear breathable, cotton clothing",
            "Use separate towels for affected areas",
            "Avoid scratching to prevent spreading",
            "Wash hands thoroughly after touching affected area",
            "Treat all infected areas simultaneously"
        ],
        "when_to_see_doctor": "If the infection doesn't improve after 2 weeks of self-care, if it's spreading, or if you have diabetes or a weakened immune system."
    },
    "Melanocytic_Nevi": {
        "name": "Melanocytic Nevi (Moles)",
        "description": "Melanocytic nevi are common moles formed by pigment cells and are usually benign. They can appear anywhere on the body and are generally harmless.",
        "symptoms": [
            "Round or oval shape",
            "Even coloring (brown, tan, black)",
            "Size usually less than 6mm",
            "Flat or slightly raised",
            "Usually smooth surface",
            "May have hair growing from them"
        ],
        "risk_factors": [
            "Fair skin",
            "Sun exposure",
            "Family history of moles",
            "Having many moles (50+)",
            "History of sunburns"
        ],
        "recommendations": [
            "Perform monthly skin self-examinations",
            "Use ABCDE criteria to monitor: Asymmetry, Border, Color, Diameter, Evolution",
            "Protect skin from sun with sunscreen and clothing",
            "Avoid tanning beds",
            "Photograph moles to track changes",
            "Have dermatologist examine new or changing moles",
            "Don't remove moles unless recommended by doctor"
        ],
        "when_to_see_doctor": "Immediately if you notice any ABCDE changes: Asymmetry, irregular Borders, multiple Colors, Diameter larger than 6mm, or Evolution/changing moles."
    },
    "Melanoma": {
        "name": "Melanoma",
        "description": "Melanoma is a serious form of skin cancer that develops in pigment-producing cells. It can spread to other organs if not caught early, making early detection crucial.",
        "symptoms": [
            "New mole or spot on skin",
            "Change in existing mole (size, shape, color)",
            "Irregular, notched, or blurred borders",
            "Multiple colors in one lesion",
            "Size larger than 6mm",
            "Itching, bleeding, or painful moles"
        ],
        "risk_factors": [
            "History of sunburns, especially in childhood",
            "Fair skin, freckles, light hair",
            "Family history of melanoma",
            "Many moles or unusual moles",
            "Weakened immune system",
            "Excessive UV exposure (sun or tanning beds)"
        ],
        "recommendations": [
            "SEE A DERMATOLOGIST IMMEDIATELY",
            "Perform monthly skin self-examinations",
            "Use broad-spectrum SPF 30+ sunscreen daily",
            "Avoid peak sun hours (10am-4pm)",
            "Wear protective clothing, hats, sunglasses",
            "Never use tanning beds",
            "Keep track of all moles with photos",
            "Report any changes to dermatologist immediately"
        ],
        "when_to_see_doctor": "URGENT - Melanoma requires immediate medical attention. See a dermatologist as soon as possible for any suspicious lesion."
    },
    "Psoriasis": {
        "name": "Psoriasis",
        "description": "Psoriasis is a chronic immune-mediated skin condition that often causes thick, scaly, and inflamed patches. It accelerates skin cell turnover, leading to rapid buildup.",
        "symptoms": [
            "Red patches with silvery scales",
            "Dry, cracked skin that may bleed",
            "Thickened, pitted nails",
            "Stiff and swollen joints",
            "Small scaling spots",
            "Itching and burning sensations"
        ],
        "risk_factors": [
            "Family history of psoriasis",
            "Stress and emotional factors",
            "Infections (strepp throat)",
            "Obesity",
            "Smoking",
            "Certain medications"
        ],
        "recommendations": [
            "Moisturize skin frequently",
            "Use medicated shampoos for scalp psoriasis",
            "Take daily baths with colloidal oatmeal",
            "Expose skin to moderate sunlight",
            "Avoid triggers (stress, infections, certain foods)",
            "Maintain healthy weight",
            "Quit smoking and limit alcohol",
            "Follow prescribed treatment plans from dermatologist"
        ],
        "when_to_see_doctor": "If the condition is painful, affecting quality of life, or not improving with self-care. Also see doctor if joints become painful or stiff."
    },
    "Seborrheic_Keratosis": {
        "name": "Seborrheic Keratosis",
        "description": "Seborrheic keratosis is a common non-cancerous skin growth that can look waxy, raised, or stuck onto the skin. They're often called 'barnacles' or 'wisdom spots.'",
        "symptoms": [
            "Waxy, stuck-on appearance",
            "Varied colors (tan, brown, black, white)",
            "Rough or grainy texture",
            "Varies from small to 2+ inches",
            "Often on face, chest, shoulders, back",
            "May be itchy if irritated"
        ],
        "risk_factors": [
            "Age (more common after age 50)",
            "Family history",
            "Sun exposure",
            "Fair skin type"
        ],
        "recommendations": [
            "No treatment needed unless bothering you",
            "Don't pick or scratch the growths",
            "Keep area moisturized",
            "Use gentle skincare products",
            "Protect from sun exposure",
            "Consult dermatologist for removal if desired (cosmetic) or if irritated"
        ],
        "when_to_see_doctor": "If the growth changes, bleeds, itches, or if you're unsure whether it's skin cancer."
    },
    "Vascular_Lesions": {
        "name": "Vascular Lesions",
        "description": "Vascular lesions are skin abnormalities related to blood vessels and can appear red, purple, or blue. Includes hemangiomas, port wine stains, and spider angiomas.",
        "symptoms": [
            "Red, purple, or blue discoloration",
            "Flat or raised lesions",
            "Spider-like patterns (spider angiomas)",
            "Birthmarks (port wine stains)",
            "Small red dots (cherry angiomas)",
            "May change with temperature or pressure"
        ],
        "risk_factors": [
            "Genetic factors",
            "Sun exposure (can worsen)",
            "Pregnancy (may appear/ worsen)",
            "Liver disease (spider angiomas)",
            "Age (cherry angiomas more common with age)"
        ],
        "recommendations": [
            "Protect from sun to prevent darkening",
            "Use concealer if concerned about appearance",
            "Monitor for changes in size or color",
            "Consult dermatologist for treatment options (laser therapy available)",
            "Birthmarks may be monitored or treated with laser",
            "Cherry angiomas generally harmless - no treatment needed"
        ],
        "when_to_see_doctor": "If the lesion changes significantly, bleeds, or if you're concerned about appearance and want treatment options."
    },
    "Viral_Infections": {
        "name": "Viral Skin Infections",
        "description": "Viral skin infections are caused by viruses and may appear as bumps, blisters, warts, or irritated lesions. Common types include HPV (warts), herpes, and molluscum contagiosum.",
        "symptoms": [
            "Skin bumps or warts",
            "Blisters or vesicles",
            "Red or pink patches",
            "Itching or burning",
            "Pain or discomfort",
            "Cluster of small bumps"
        ],
        "risk_factors": [
            "Direct contact with infected person",
            "Weakened immune system",
            "Skin breaks or cuts",
            "Shared towels or surfaces",
            "Sexual contact (certain viruses)",
            "Children (more common)"
        ],
        "recommendations": [
            "Keep affected area clean and dry",
            "Don't touch or pick at lesions",
            "Don't share personal items (towels, razors)",
            "Wash hands frequently",
            "Use over-the-counter wart treatments if applicable",
            "Keep warts covered to prevent spread",
            "Boost immune system with healthy lifestyle",
            "Some viral infections resolve on their own"
        ],
        "when_to_see_doctor": "If lesions don't improve, are spreading, painful, or if you're unsure of the diagnosis."
    }
}

# Simple fallback descriptions for backwards compatibility
DISEASE_FALLBACK_DESCRIPTIONS = {key: value["description"] for key, value in DISEASE_CATALOG.items()}

def get_disease_info(disease_name: str) -> dict:
    """Get comprehensive disease information"""
    return DISEASE_CATALOG.get(disease_name, {
        "name": disease_name.replace("_", " "),
        "description": "This condition requires professional medical evaluation.",
        "symptoms": ["Information not available"],
        "risk_factors": ["Information not available"],
        "recommendations": ["Consult a healthcare professional for proper diagnosis and treatment"],
        "when_to_see_doctor": "Consult a healthcare professional"
    })
