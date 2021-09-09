import glob

class ConllReader:
    def read(self, file_path: str, samples_number=None):
        tokens = []
        tags = []
        with open(file_path) as file:
            lines = file.readlines()
            sample_i = 0
            for line in lines:
                line = line.strip(' \n')
                if not line:
                    if tokens and tags:
                        assert len(tokens) == len(tags)
                        yield (tokens, tags)
                        sample_i += 1
                        if samples_number and sample_i >= samples_number:
                            break
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