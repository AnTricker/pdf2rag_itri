from __future__ import annotations

from local_rag.models import CorpusProfile, KnowledgeRecord


class FakeCorpusProfileBackend:
    name = "fixture-profile"
    model_name = "fixture-llm"
    prompt_version = "corpus-profile-v1"

    def build(
        self,
        *,
        document_name: str,
        document_id: str,
        records: list[KnowledgeRecord],
    ) -> CorpusProfile:
        return CorpusProfile(
            document_id=document_id,
            document_name=document_name,
            document_type="vendor manual",
            summary="Equipment operation and reset procedures.",
            in_scope_topics=["operation", "reset"],
            out_of_scope_examples=["weather"],
            key_sections=["Reset procedure"],
            key_entities=["controller"],
            profile_source=self.name,
            generation_model=self.model_name,
            prompt_version=self.prompt_version,
        )
