from modules.invblock import INV_block
from torch import nn

class Hinet(nn.Module):

    def __init__(self, in_1=3, in_2=3):
        super(Hinet, self).__init__()

        self.inv1 = INV_block(in_1=in_1, in_2=in_2)
        self.inv2 = INV_block(in_1=in_1, in_2=in_2)
        self.inv3 = INV_block(in_1=in_1, in_2=in_2)
        self.inv4 = INV_block(in_1=in_1, in_2=in_2)
        self.inv5 = INV_block(in_1=in_1, in_2=in_2)
        self.inv6 = INV_block(in_1=in_1, in_2=in_2)
        self.inv7 = INV_block(in_1=in_1, in_2=in_2)
        self.inv8 = INV_block(in_1=in_1, in_2=in_2)

        self.inv9 = INV_block(in_1=in_1, in_2=in_2)
        self.inv10 = INV_block(in_1=in_1, in_2=in_2)
        self.inv11 = INV_block(in_1=in_1, in_2=in_2)
        self.inv12 = INV_block(in_1=in_1, in_2=in_2)
        self.inv13 = INV_block(in_1=in_1, in_2=in_2)
        self.inv14 = INV_block(in_1=in_1, in_2=in_2)
        self.inv15 = INV_block(in_1=in_1, in_2=in_2)
        self.inv16 = INV_block(in_1=in_1, in_2=in_2)

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out


