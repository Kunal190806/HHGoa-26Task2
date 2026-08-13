import json
import os

def create_mock_dataset():
    data = []
    
    # 20 simple mock records matching MSMARCO-XI structure exactly
    for i in range(20):
        record = {
            "query_id": f"mock_{i}",
            "target_lang": "hi" if i % 2 == 0 else "mr",
            "query": f"यह एक मॉक प्रश्न है {i}?",
            "Eng_Query": f"This is a mock query {i}?",
            "Answer": f"मॉक उत्तर {i} यहाँ है।",
            "Eng_Answer": f"The mock answer {i} is here.",
            "passages": {
                "English_passages": [
                    f"This is the correct passage for query {i}. It contains the answer.",
                    f"This is a distractor passage for query {i}.",
                    f"Another irrelevant passage for {i}."
                ],
                "Translated_passages": [
                    f"यह प्रश्न {i} के लिए सही मार्ग है। इसमें उत्तर है।",
                    f"यह प्रश्न {i} के लिए एक भटकाने वाला मार्ग है।",
                    f"{i} के लिए एक और अप्रासंगिक मार्ग।"
                ],
                "is_selected": [1, 0, 0]
            }
        }
        data.append(record)
        
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    with open("mock_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("Mock dataset created at data/mock_dataset.json")

if __name__ == "__main__":
    create_mock_dataset()
