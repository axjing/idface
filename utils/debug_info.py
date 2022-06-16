class PeopleInfo(object):
    _defaults = {
        "Name": "two dog",
        "ID": "111111111",
        "Address": "Kingdom Come",
        "Tel": 88888888,
        "Male": "Boy",
    }

    def __init__(self):
        self.Name = self._defaults["Name"]
        self.ID = self._defaults["ID"]
        self.Address = self._defaults["Address"]
        self.Tel = self._defaults["Tel"]
        self.Male = self._defaults["Male"]
        print(self.Name)
class PeopleInfo1():
    _defaults = {
        "Name": "two dog",
        "ID": "111111111",
        "Address": "Kingdom Come",
        "Tel": 88888888,
        "Male": "Boy",
    }


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        print(self.Name)
if __name__ == "__mian__":
    dog2 = PeopleInfo()
    dog2.__init__()
