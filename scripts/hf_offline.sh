# source edilir (çalıştırılmaz): `source scripts/hf_offline.sh`
# Önbellek TAM ise (scripts/predownload.py işaret dosyasını yazdıysa) HF'ye hiç
# istek atmadan çalış. Kapatmak için: HF_OFFLINE_MODE=off
if [ "${HF_OFFLINE_MODE:-auto}" != "off" ] && [ -f results/.hf_offline_ok ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  echo "HF çevrimdışı mod: AÇIK (önbellek tam; Hub'a istek yok)"
else
  echo "HF çevrimdışı mod: kapalı (önbellek tam değil ya da HF_OFFLINE_MODE=off)"
fi
