import spacetime as st

f = st.Frame()

f.append(st.Worldline([[0, 1, 0]], ends_vel_s=[0.1, 0]), 'observer')
f.append(st.Worldline([
    [0, 0, 0],
    [2, 1, 0],
    [4, 0, 1],
    [6, -1, 0],
    [8, 0, -1],
    ], ends_vel_s=[0, 0]))

sim = st.ObserverSim(f)

import pygame
