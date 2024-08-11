function Fy = Fy_ChanSandu(slip, Fz)

    Fy = (((1./(1+0.9068.^(-(slip*180/pi))))-0.5)).*(2.2323.*Fz);

end