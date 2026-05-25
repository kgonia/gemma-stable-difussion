# StyleJourney v10 compatibility fix for transformers >= 5.x
# In transformers 5.x, CLIPTextModel no longer has `text_model` submodule.
# Diffusers 0.37.x `from_single_file` expects `model.text_model.embeddings`.
#
# Patch: /gemma-stable-difussion/.venv/lib/python3.13/site-packages/diffusers/loaders/single_file_utils.py:1702
#
# Was:
#   position_embedding_dim = model.text_model.embeddings.position_embedding.weight.shape[-1]
#
# Now:
#   if hasattr(model, 'text_model'):
#       position_embedding_dim = model.text_model.embeddings.position_embedding.weight.shape[-1]
#   else:
#       position_embedding_dim = model.embeddings.position_embedding.weight.shape[-1]
