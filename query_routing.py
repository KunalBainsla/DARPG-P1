import torch
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
import json

# Open the JSON file
with open('/home/shiroyasha/ML/final_darpg_proj/label_dict_inverse.json', 'r') as file:
    # Load the content of the file and convert it to a dictionary
    label_dict_inverse = json.load(file)

# Initialize the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tmp_model = BertForSequenceClassification.from_pretrained("Shiroyaksha/model_d",
                                                      num_labels=len(label_dict_inverse),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

def query_routing_page():
    st.title("Query Routing")
    query = st.text_area("Enter your query:")
    if st.button("Route Query"):
        inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = tmp_model(**inputs)

        # Extract the logits
        logits = outputs.logits
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Get the predicted class label
        predicted_class = torch.argmax(probs, dim=-1).item()
        predicted_class_label = label_dict_inverse[str(predicted_class)]

        st.write("This query should be routed to:", predicted_class_label)

query_routing_page()
