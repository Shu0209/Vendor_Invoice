# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px


# from inference.predict_freight import predict_freight_cost
# from inference.predict_invoice_flag import predict_invoice_flag


# st.set_page_config(
#     page_title="Vendor Invoice Intelligence Portal",
#     page_icon="sss",
#     layout="wide"
# )




# st.markdown("""
# # Vendor Invoice Portal
# ### AI-Driven freight Cost Prediction & Invoice Risk Flagging
            
# This internal analytics portal leverages machine to 
# - **Forecast cost accurately**
# - **Dectect risk or abnormal vendor invoice**
# - **Reduce financial leakage and manual workload**
# """)


# # SideBar


# st.sidebar.title("Model Selection")
# selected_model=st.sidebar.radio(
#     "Choose Prediction Module",
#     [
#         "Freight Cost Prediction",
#         "Invoice Manual Approval Flag"
#     ]
# )

# st.sidebar.markdown(
#     """
# ---
# **Business Impact**
# - Improve cost forecasting
# - Reduce invoice fraud & anomalies
# - Faster finance operations
# """
# )


# # Freight Cost Prediction


# if selected_model=="Freight Cost Prediction":
#     st.subheader("Freight Cost Prediction")

#     st.markdown("""
#                 ** Objective **
#                 Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**
#                 to support budgeting, forcasting, and vendor negotiations.                
#                 """)
    
#     with st.form("freight_form"):
#         col1, col2=st.columns(2)

#         with col1:
#             quantity=st.number_input(
#                "Quantity",
#                min_value=1,
#                value=1200 
#             )


#         with col2:
#             dollars=st.number_input(
#                 "Invoice Dollars",
#                 min_value=1.0,
#                 value=18500.0
#             )
        
#         submit_freight=st.form_submit_button("Predict Freight Cost")

#     if submit_freight:
#         input_data={
#             "Quantity":[quantity],
#             "Dollars":[dollars]
#         }

#         prediction=predict_freight_cost(input_data)['Predicted_Freight']

#         st.success("Prediction completed successfully.")

#         st.metric(
#             label="Estimated Freight Cost",
#             value=f"${prediction[0]:,.2f}"

#         )
#     # Invoice Flag

# else:
#     st.subheader("Invoice Manual Approval Prediction")

#     st.markdown("""
#                     **Objective:**
#                     Predict whether a vendor invoice should br **flagged for manual approval**
#                     based on abnormal cost, freight, or delivery patterns.
#                     """)
        
#     with st.form("Invoice_flag_form"):
#         col1, col2, col3=st.columns(3)

#         with col1:
#             invoice_quantity=st.number_input(
#                 "Invoice Quantity",
#                 min_value=1,
#                 value=50
#                 )
#             freight=st.number_input(
#                 "Freight Cost",
#                 min_value=0.0,
#                 value=1.73
#                 )

#         with col2:
#             invoice_dollars=st.number_input(
#                 "Invoice Dollars",
#                 min_value=1.0,
#                 value=352.95
#                 )
#             total_item_quantity=st.number_input(
#                 "Total Item Quantity",
#                 min_value=1,
#                 value=162
#                 )

#         with col3:
#             total_item_dollars=st.number_input(
#                 "Total Item Dollars",
#                 min_value=1.0,
#                 value=2476.0
#                 )
            
#         submit_flag=st.form_submit_button("Evaluate Invoice Risk")
#     if submit_flag:
#         input_data={
#             "invoice_quantity":[invoice_quantity],
#             "invoice_dollars":[invoice_dollars],
#             "freight":[freight],
#             "total_item_quantity":[total_item_quantity],
#             "total_item_dollars":[total_item_dollars]
#         }

#         flag_prediction=predict_invoice_flag(input_data)['Predicted_flag']

#         is_flagged=bool(flag_prediction[0])


#         if is_flagged:
#             st.error("Invoice requires **MANUAL APPROVAL**")
#         else:
#             st.success("Invoice is **SAFE for Auto-Approval**")






import streamlit as st
import pandas as pd

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)

# ==============================
# Header
# ==============================
st.markdown("""
# Vendor Invoice Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast cost accurately**
- **Detect risk or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

# ==============================
# Sidebar
# ==============================
st.sidebar.title("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**
- Improve cost forecasting  
- Reduce invoice fraud & anomalies  
- Faster finance operations  
""")

# ==============================
# Freight Cost Prediction
# ==============================
if selected_model == "Freight Cost Prediction":

    st.subheader("Freight Cost Prediction")

    st.markdown("""
**Objective:**
Predict freight cost using **Quantity** and **Invoice Dollars**
to support budgeting, forecasting, and vendor negotiations.
""")

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input("Quantity", min_value=1, value=1200)

        with col2:
            dollars = st.number_input("Invoice Dollars", min_value=1.0, value=18500.0)

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        try:
            input_data = {
                "Dollars": [dollars],
                "Quantity": [quantity]
            }

            prediction_df = predict_freight_cost(input_data)
            prediction = prediction_df["Predict_Freight"][0]

            st.success("Prediction completed successfully.")

            st.metric(
                label="Estimated Freight Cost",
                value=f"${prediction:,.2f}"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==============================
# Invoice Flag Prediction
# ==============================
else:

    st.subheader("Invoice Manual Approval Prediction")

    st.markdown("""
**Objective:**
Predict whether a vendor invoice should be **flagged for manual approval**
based on abnormal cost, freight, or delivery patterns.
""")

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=1.73)

        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=352.95)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=162)

        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=2476.0)

        submit_flag = st.form_submit_button("Evaluate Invoice Risk")

    if submit_flag:
        try:
            input_data = {
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars]
            }

            flag_df = predict_invoice_flag(input_data)
            flag_prediction = flag_df["Predict_Flag"][0]

            if bool(flag_prediction):
                st.error("🚨 Invoice requires **MANUAL APPROVAL**")
            else:
                st.success("✅ Invoice is **SAFE for Auto-Approval**")

        except Exception as e:
            st.error(f"Error: {str(e)}")