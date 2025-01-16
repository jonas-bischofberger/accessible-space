from accessible_space.tests.test_real_world_data import test_real_world_data

import matplotlib.pyplot as plt
import streamlit as st

st.write(plt.style.available)

style = "seaborn-v0_8"

for style in plt.style.available:
    plt.style.use(style)

    plt.figure()
    plt.title(style)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.plot([1, 2, 3, 4], [1, 2, 3, 4])
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.show()
    st.write(plt.gcf())
    plt.close()

    # test_real_world_data(1)
