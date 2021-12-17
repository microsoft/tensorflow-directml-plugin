import absl.testing.absltest as absltest

class GoodTest(absltest.TestCase):
    def testA(self):
        self.assertFalse(False)

    def testB(self):
        self.assertFalse(False)

class AnotherGoodTest(absltest.TestCase):
    def testA(self):
        self.assertFalse(False)

    def testB(self):
        self.assertFalse(False)

if __name__ == "__main__":
    absltest.main()