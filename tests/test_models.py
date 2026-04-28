import unittest





class TestGAMisFunctional(unittest.TestCase):
    def test_forward_pass():
        pass

    def test_forward_shapes_pass():
        pass






class TestGAMcompression(unittest.TestCase):
    def test_compressed(self):
        D = 10
        C = 1
        sizes = [-1, 256, 128, 64, -1]
        sizes[0] = D; sizes[-1] = C;
        small_sizes = [-1, 32, 24, 16, -1]
        small_sizes[0] = D; small_sizes[-1] = C;
        FIS_interactions = [(0,), (1,), (2,),]

        for GAMModel in [Smooth_Or_MNIST_SIAN]:
            gam = GAMModel(sizes, FIS_interactions)

            N = 1000
            X = torch.randn(N,D)

            out = gam(X)
            gam.compress()
            out2 = gam(X)
            gam.blocksparse()
            out3 = gam(X)

            self.assertEqual(out, out2)
            self.assertEqual(out, out3)



class TestEmptyIndexSet():
    def test():

        #TEST WITH OTHERS

        #TEST WITHOUT OTHERS (ONLY A BIAS)