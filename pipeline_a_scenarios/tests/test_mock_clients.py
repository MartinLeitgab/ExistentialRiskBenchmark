class MockAnthropicClient:
    class messages:
        @staticmethod
        def create(**kwargs):
            class R:
                model = "mock-anthropic-model"
                content = [type("T", (), {"text": "anthropic mock"})()]
                usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()
            return R()

    class batches:
        @staticmethod
        def create(requests):
            return type("B", (), {"id": "anth-batch-1"})()

        @staticmethod
        def retrieve(batch_id):
            class Batch:
                class request_counts:
                    processing = 0
                    succeeded = 1
                    errored = 0
            return Batch()

        @staticmethod
        def results(batch_id):
            class Result:
                custom_id = "req-1"
                result = type(
                    "RR",
                    (),
                    {
                        "type": "succeeded",
                        "message": type(
                            "M",
                            (),
                            {"content": [type("T", (), {"text": "anthropic batch"})()]},
                        ),
                    },
                )()
            yield Result()



class MockOpenAIClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return type(
                    "R",
                    (),
                    {
                        "model": "mock-openai-model",
                        "choices": [
                            type(
                                "C",
                                (),
                                {
                                    "message": type(
                                        "M",
                                        (),
                                        {"content": "openai mock"},
                                    )(),
                                },
                            )
                        ],
                        "usage": type(
                            "U",
                            (),
                            {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                            },
                        )(),
                    },
                )()


    class files:
        @staticmethod
        def create(**kwargs):
            return type("F", (), {"id": "file-1"})

        @staticmethod
        def content(file_id):
            return (
                b'{"custom_id":"req-1","response":{"choices":[{"message":{"content":"openai batch"}}]}}'
            )

    class batches:
        @staticmethod
        def create(**kwargs):
            return type("B", (), {"id": "openai-batch-1"})

        @staticmethod
        def retrieve(batch_id):
            return type(
                "BR",
                (),
                {
                    "status": "completed",
                    "output_file_id": "file-1",
                },
            )



class MockGeminiClient:
    class models:
        @staticmethod
        def generate_content(**kwargs):
            return type("R", (), {"text": "gemini mock"})

    class files:
        @staticmethod
        def upload(**kwargs):
            return type("F", (), {"name": "gemini-file-1"})()

        @staticmethod
        def get(name):
            # Return an object with a .state attribute
            return type("FS", (), {"state": "SUCCEEDED"})()

        @staticmethod
        def download(name):
            return (
                b'{"key":"req-1","response":{"candidates":[{"content":{"parts":[{"text":"gemini batch"}]}}]}}'
            )
