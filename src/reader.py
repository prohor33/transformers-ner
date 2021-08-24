class ConllReader:
    def read(self, file_path: str):
        tokens = []
        tags = []
        
        with open(file_path) as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip(' \n')
                if not line:
                    if tokens and tags:
                        assert len(tokens) == len(tags)
                        yield (tokens, tags)
                        tokens = []
                        tags = []
                    continue
                if line.startswith("-DOCSTART-"):
                    continue
                cols = line.split()
                assert len(cols) > 1, f"\"{line}\", {cols}"
                token = cols[0]
                tag = cols[-1]
                tokens.append(token)
                tags.append(tag)