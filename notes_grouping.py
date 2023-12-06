# Databricks notebook source
top_notes_mapping = {
    "Floral": [
        "Black Rose",
        "Osmanthus",
        "Fig Tree",
        "Lotus",
        "Tobacco Blossom",
        "Rose",
        "Iris",
        "Damask Rose",
        "Jasmine",
        "Bulgarian Rose",
        "Heliotrope",
        "Almond",
        "French labdanum",
        "Cashmeran",
        "Moss",
        "Woody Notes",
        "Woodsy Notes",
        "Ambergris",
        "White Musk",
    ],
    "Gourmand": [
        "Rhubarb",
        "Mexican chocolate",
        "Vanilla Bean",
        "Cotton Candy",
        "White Honey",
        "Milk",
        "Coconut",
        "Black locust Honey",
        "Dark Chocolate",
        "Licorice",
        "White Chocolate",
        "Marshmallow",
        "Caramel",
        "Brown sugar",
        "Vanilla",
        "Tonka Bean",
        "Almond",
        "Caramel",
    ],
    "Woody": [
        "Australian Sandalwood",
        "Oak",
        "Oakmoss",
        "Cedarwood",
        "Brazilian Rosewood",
        "Cashmirwood",
        "Palisander Rosewood",
        "White Woods",
        "Ebony",
        "White Wood",
        "Mahogany",
        "Cedar Essence",
        "Virginian Cedar",
        "Cashmere Wood",
        "Woodsy Notes",
        "Woody Notes",
        "Sandalwood",
        "Amberwood",
        "Vetiver",
        "Patchouli",
        "Amber",
        "Cedar",
        "Benzoin",
    ],
    "Spicy": [
        "Cinnamon",
        "Spices",
        "Saffron",
        "Cloves",
        "Pink Pepper",
        "Powdery Notes",
        "Coumarin",
        "Siam Benzoin",
    ],
    "Earthy": [
        "Frankincense",
        "Seaweed",
        "Incense",
        "Iso E Super",
        "Carrot",
        "Sage",
        "Styrax",
        "Oakmoss",
        "Bran",
        "Earthy Notes",
        "Orris",
        "Papyrus",
        "Vetiver",
        "Moss",
    ],
    "Fruity": ["Raspberry", "Fig", "Peach", "Blackberry"],
    "Resinous": [
        "Frankincense",
        "Guaiac Wood",
        "Agarwood (Oud)",
        "Tolu Balsam",
        "Spanish Labdanum",
        "Resins",
        "Resin",
        "Olibanum",
        "Amber",
        "Ambergris",
        "Benzoin",
    ],
    "Musky/Animalic": [
        "Gray Musk",
        "Animal notes",
        "Civet",
        "Ambrette (Musk Mallow)",
        "Ambrettolide",
        "Ambrofix™",
        "Musk",
        "Tobacco",
        "Tobacco Leaf",
    ],
    "Herbal/Green": [
        "Matcha Tea",
        "Sage",
        "Galbanum",
        "Eucalyptus",
        "Rosemary",
        "Fir",
        "Green Notes",
        "Green Accord",
    ],
    "Citrus": ["Tea", "Mandarin", "Bergamot", "Lemon"],
    "Oriental/Exotic": [
        "Laotian Oud",
        "Argan",
        "Mate",
        "Peru Balsam",
        "Kyara Incense",
        "Labdanum",
        "Ambreine",
        "Indonesian Patchouli Leaf",
        "Akigalawood",
        "Orris",
        "Geranium",
        "Marjoram",
        "Orris",
        "Ylang-Ylang",
        "Singapore Patchouli",
    ],
}

# COMMAND ----------

middle_notes_mapping = {
    "Floral": [
        "Pink Violet",
        "Lily of the Valley",
        "Wisteria",
        "Chinese Osmanthus",
        "Violet Leaf",
        "Acacia",
        "Moroccan Rose",
        "Lotus",
        "White Tea",
        "Honeysuckle",
        "Rose Petals",
        "Campion Flower",
        "Cyclamen",
        "Water Lily",
        "Black Orchid",
        "Jasmine Sambac",
        "Thistle",
        "Jasmine",
        "Silk Tree Blossom",
        "Rose Oil",
        "White Flowers",
        "Magnolia",
        "Artemisia",
        "Osmanthus",
        "Timur",
        "Snowdrops",
        "Ivy",
        "Poppy",
        "Cornflower or Sultan Seeds",
        "Wisteria Flower",
        "Taif Rose",
    ],
    "Citrus/Fruity": [
        "Clementine",
        "Quince",
        "Bergamot",
        "Lemon",
        "Grapefruit",
        "Amalfi Lemon",
        "Sicilian Mandarin",
        "Citron",
        "Raspberry Leaf",
        "Kumquat",
        "Sour Cherry",
        "Grapefruit",
        "Calabrian Bergamot",
        "Tunisian Orange Blossom",
        "Italian Lemon",
        "Citrus Leaves",
        "Lemon",
        "Grapefruit",
        "Lemon",
        "Lime",
    ],
    "Woody": [
        "Sandalwood",
        "Moss",
        "Pine Tree",
        "Pine Needles",
        "Cedar",
        "Virginian Cedar",
        "Palo Santo",
        "Oak",
        "Oakmoss",
        "Brazilian Rosewood",
        "White Wood",
        "Cypress",
        "Atlas Cedar",
        "Guaiac Wood",
        "Oakmoss",
        "Oak",
    ],
    "Spicy": [
        "Vanilla",
        "Cognac",
        "Rosemary",
        "Coriander",
        "Juniper Berries",
        "Thyme",
        "Caraway",
        "Cannabis",
        "Nutmeg",
        "Anise",
        "Cloves",
        "Fennel",
        "Star Anise",
        "Cinnamon Clove",
        "Spices",
        "Clove",
        "Spices",
        "Cardamom",
    ],
    "Gourmand": [
        "Vanilla",
        "Almond Blossom",
        "Dark Chocolate",
        "Cacao",
        "Praline",
        "Hazelnut",
        "Chocolate",
        "Caramel",
        "Pistachio",
        "Vanilla",
        "Milk",
        "Fig",
        "Coconut Milk",
        "Coconut",
        "Dried Plum",
        "Oat",
        "Bourbon Vanilla",
        "Timur",
        "White Suede",
        "Truffle",
        "Caramel",
        "Vanilla",
        "Fig",
        "Chocolate",
        "Hazelnut",
        "Chocolate",
        "Caramel",
        "Pistachio",
        "Vanilla Coconut",
        "Dried Plum",
        "Oat",
        "Bourbon Vanilla",
        "Chocolate",
        "White Suede",
        "Truffle",
        "Caramel",
    ],
    "Fresh/Aquatic": [
        "Salt",
        "Rain Notes",
        "Watery Notes",
        "Water Notes",
        "Sea Notes",
        "Dew Drop",
        "Rain Notes",
        "Sea Salt",
        "Solar Notes",
        "Sea Notes",
        "Snowdrops",
        "Water Lily",
    ],
    "Green/Herbal": [
        "Clary Sage",
        "Broom",
        "Rosemary",
        "Eucalyptus",
        "Verbena",
        "Lemongrass",
        "Nettle",
        "Thyme",
        "Basil",
        "Green Accord",
        "Myrtle",
        "Mint",
    ],
    "Earthy/Spicy": [
        "Sandalwood",
        "Benzoin",
        "Moss",
        "Galbanum",
        "Juniper Berries",
        "Myrrh",
        "Papyrus",
        "Carrot Seeds",
        "Vetiver",
        "Patchouli",
    ],
}

# COMMAND ----------

base_notes_mapping = {
    "Floral": [
        "African Orange Flower",
        "Apricot Blossom",
        "Artemisia",
        "Bellflower",
        "Bulgarian Rose",
        "Cherry Blossom",
        "Chinese Peony",
        "Chinese Plum",
        "Damask Rose",
        "Freesia",
        "Gardenia",
        "Geranium",
        "Hibiscus",
        "Hyacinth",
        "Jasmine",
        "Jasmine Sambac",
        "Lily",
        "Lily-of-the-Valley",
        "Lilac",
        "Magnolia",
        "Magnolia Petals",
        "Marigold",
        "Moon Flower",
        "Narcissus",
        "Neroli",
        "Neroli Essence",
        "Orange Blossom",
        "Osmanthus",
        "Orange Blossom",
        "Peony",
        "Rose",
        "Pomegranate Blossom",
        "Rose",
        "Syringa",
        "Tunisian Neroli",
        "Tunisian Orange Blossom",
        "Turkey Red Rose",
        "Turkish Rose",
        "Violet",
        "Violet Leaf",
        "White Flowers",
        "White Rose",
        "Ylang-Ylang",
        "Pink Freesia",
    ],
    "Citrus": [
        "Amalfi Lemon",
        "Bergamot",
        "Calabrian Mandarin",
        "Calabrian bergamot",
        "California Orange",
        "Californian Lemon",
        "Citron",
        "Clementine",
        "Grapefruit",
        "Italian Lemon",
        "Italian Mandarin",
        "Lemon",
        "Lemon Blossom",
        "Lemon Leaf",
        "Lemon Verbena",
        "Lemon Zest",
        "Lime",
        "Mandarin Blossom",
        "Mandarin Orange",
        "Nashi Pear",
        "Orange",
        "Sicilian Lemon",
        "Sicilian Mandarin",
        "Sweet Orange",
        "Tangerine",
        "Yellow Mandarin",
        "Yuzu",
        "Lemongrass",
    ],
    "Fruity": [
        "Almond",
        "Apple",
        "Apricot",
        "Black Cherry",
        "Black Currant",
        "Blackcurrant Syrup",
        "Blood Orange",
        "Cherry",
        "Coconut",
        "Fig",
        "Fig Nectar",
        "Kiwi",
        "Litchi",
        "Melon",
        "Nashi Pear",
        "Papaya",
        "Peach",
        "Pear",
        "Pineapple",
        "Plum",
        "Raspberry",
        "Red Apple",
        "Red Berries",
        "Red Currant",
        "Rhubarb",
        "Sour Cherry",
        "Strawberry",
        "Watermelon",
        "Yellow Fruits",
    ],
    "Spicy": [
        "Aldehydes",
        "Black Cardamom",
        "Black Pepper",
        "Ceylon Cinnamon",
        "Clove",
        "Coriander",
        "Coriander Leaf",
        "Ginger",
        "Green Pepper",
        "Guatemalan Cardamom",
        "Juniper",
        "Juniper Berries",
        "Mate",
        "Mint",
        "Nigerian Ginger",
        "Nutmeg",
        "Pepper",
        "Pink Pepper",
        "Sichuan Pepper",
        "Star Anise",
        "Turmeric",
        "Ginger",
    ],
    "Green/Herbal": [
        "Angelica",
        "Bamboo",
        "Basil",
        "Birch Leaf",
        "Calamansi",
        "Caraway",
        "Cassia",
        "Cyclamen",
        "Fig Leaf",
        "Green Almond",
        "Green Leaves",
        "Green Mandarin",
        "Green Notes",
        "Green Tea",
        "Hedione",
        "Lavender",
        "Lavender Extract",
        "Mint",
        "Myrtle",
        "Parsley",
        "Petitgrain",
        "Pineapple Leaf",
        "Rosemary",
        "Sage",
        "Shiso",
        "Tomato Leaf",
        "Tarragon",
        "Tea",
        "Thyme",
        "Verbena",
        "Bergamot",
        "Saffron",
    ],
    "Gourmand": [
        "Cacao",
        "Cherry Liqueur",
        "Coconut",
        "Coffee",
        "Cognac",
        "Honey",
        "Honeysuckle",
        "Lemon Zest",
        "Limoncello",
        "Pistachio",
        "Rum",
        "Caramel",
    ],
    "Woody": [
        "Cedar",
        "Elemi resin",
        "Fig",
        "Myrrh",
        "Palo Santo",
        "Sandalwood",
        "Virginia Cedar",
    ],
    "Bitter/Aromatic": [
        "Bitter Almond",
        "Bitter Orange",
        "Calone",
        "Carrot Seeds",
        "Davana",
        "Galbanum",
        "Incense",
        "Labdanum",
    ],
    "Earthy": ["Vetiver", "Myrica", "Patchouli", "Truffle"],
    "Aquatic": [
        "Dew Drop",
        "Ozonic notes",
        "Sea Notes",
        "Sea Salt",
        "Water Notes",
        "Watery Notes",
    ],
    "Powdery": ["Iris", "Powdery Notes"],
    "Musk": ["Ambrette (Musk Mallow)", "Musk"],
    "others": ["Elemi", "Salt"],
}

# COMMAND ----------

main_accords_grouping = {
  "citrus": ["citrus"],
  "floral": ["floral", "white_floral", "rose", "lavender", "tuberose", "iris", "yellow_floral", "violet"],
  "woody": ["woody", "leather"],
  "spicy": ["warm_spicy", "fresh_spicy"],
  "vanilla": ["vanilla"],
  "powdery": ["powdery"],
  "sweet": ["sweet", "fruity", "almond", "cherry"],
  "musky": ["musky"],
  "fresh": ["fresh", "marine"],
  "earthy": ["earthy", "amber", "tobacco", "aquatic", "soapy", "green", "alcohol", "savory"]
}
