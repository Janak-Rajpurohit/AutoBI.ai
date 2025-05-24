import streamlit as st
# import streamlit_extras
# import streamlit_extras.chart_container
# import streamlit_extras.dataframe_explorer
# import streamlit_extras.echo_expander
# from streamlit_extras import card 
# import pandas as pd
# import streamlit_extras.great_tables
# from streamlit_extras import switch_page_button
# import streamlit_extras.toggle_switch
# with streamlit_extras.echo_expander.echo_expander("above"):
#     st.write("hello")
f =st.file_uploader("Upload csv")
# streamlit_extras.dataframe_explorer(pd.read_csv(f))
# df= pd.read_csv(f)
# df = streamlit_extras.dataframe_explorer.dataframe_explorer(df)  #multi column dynamic filtering
# st.dataframe(df)

# card.card(
#     title="Hello World!",
#     text="Some description",
#     image="https://plus.unsplash.com/premium_photo-1681810994162-43dbe0919d3f?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8bWFjaGluZSUyMGxlYXJuaW5nJTIwbW9kZWx8ZW58MHx8MHx8fDA%3D",
#     url="https://www.google.com",
# )
# streamlit_extras.great_tables.example()

# if st.button("switch"):
#     switch_page_button.switch_page("upload")


# streamlit_extras.toggle_switch.example()

# provide a chart  container with chart data export section
# streamlit_extras.chart_container.chart_container()

# def render():

st.title(" contact Page")
st.write("Welcome to the contact page of Adidas Sales Analytics.")
