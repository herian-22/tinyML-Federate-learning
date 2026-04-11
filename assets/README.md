# 📂 Assets — Panduan Penambahan GIF Demo

Folder ini menyimpan aset visual (GIF/gambar) untuk README proyek.

## File yang perlu ditambahkan

| File | Deskripsi |
|------|-----------|
| `demo-serial.gif` | Rekaman Serial Monitor saat firmware berjalan (tampilkan progress 7 fase) |
| `benchmark-results.gif` | Rekaman tabel ringkasan final benchmark (7 fase, akurasi & latensi) |

---

## Cara merekam GIF

### Alat yang direkomendasikan

| Platform | Tools |
|----------|-------|
| Windows | [ScreenToGif](https://www.screentogif.com/) (gratis) |
| macOS | [Kap](https://getkap.co/) atau QuickTime + FFmpeg |
| Linux | `peek`, `byzanz`, atau `ffmpeg` |

### Langkah-langkah

1. **Flash firmware** ke ESP32:
   ```bash
   idf.py -p /dev/ttyUSB0 flash monitor
   ```

2. **Rekam jendela Serial Monitor** dari awal boot hingga tabel ringkasan final muncul.

3. **Crop & compress** GIF agar ukurannya ≤ 5 MB (GitHub membatasi tampilan inline).
   - Resolusi rekomendasi: lebar 800–1000 px
   - FPS: 10–15 sudah cukup

4. **Simpan file** di folder ini:
   - `assets/demo-serial.gif`
   - `assets/benchmark-results.gif`

5. **Commit & push** file GIF ke repository.

---

> Setelah file GIF diunggah, section **🎬 Demo & Preview** di README akan otomatis menampilkan animasinya.
