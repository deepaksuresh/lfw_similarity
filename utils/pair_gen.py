import os, glob, random
class Pair(object):
    first = "first"
    second = "second"
    label = ""

    def __init__(self, src = "/home/nanonets/lfw_funneled"):
        self.everyone = self.people2path(src)
    def people2path(self, src):
        everyone = {}
        person_dir = glob.glob(os.path.join(src,"*"))
        for person in person_dir:
            person_name = os.path.basename(person)
            everyone[person_name] = glob.glob(os.path.join(*[src, person_name, "*.jpg"]))
        return everyone
    
    def get_pair(self):
        names = list(self.everyone.keys())
        #print("names ", names)
        while True:
            first = random.choice(names)
            is_same = random.random()>0.5
            if is_same:
                second = first
            else:
                second = first
                while second == first:
                    second = random.choice(names)
            img1 = random.choice(self.everyone[first])
            img2 = random.choice(self.everyone[second])
            yield ({self.first:img1, self.second:img2, self.label:is_same})
