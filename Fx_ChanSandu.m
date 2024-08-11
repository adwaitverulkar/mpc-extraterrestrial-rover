function Fx = Fx_ChanSandu(slip, Fz)

    Fx = (1./(1+0.9476.^(-(100*slip)))-0.5).*(-1.0253*Fz)+-7.9593;
end