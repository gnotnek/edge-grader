from edge_grader.rag.corpus import (
    Page,
    chunk,
    clean_text,
    drop_boilerplate,
    keep_block,
    wiki_pages,
)


def test_clean_fixes_broken_ligatures():
    assert clean_text("Identi昀椀kasi dan Bere昀氀eksi") == "Identifikasi dan Berefleksi"
    assert clean_text("Infograٴk, Sertiٶkat, Pro˪l") == "Infografik, Sertifikat, Profil"
    assert clean_text("ﬁsh and ﬂow") == "fish and flow"


def test_clean_drops_invisible_and_private_use_characters():
    assert clean_text("a\u200bb c") == "ab c"


def test_clean_joins_words_hyphenated_across_lines():
    assert clean_text("photo-\nsynthesis makes food") == "photosynthesis makes food"
    assert clean_text("well-known") == "well-known"


def test_clean_collapses_whitespace_and_bullets():
    assert clean_text("●  Animals \n sense\tlight") == "Animals sense light"


def test_keep_block_drops_captions_and_short_lines():
    assert not keep_block("Image by analogicus; pixabay.com; CC0")
    assert not keep_block("Sumber: Diana Wells/healthline.com (2017)")
    assert not keep_block("a. vitamin")
    assert not keep_block("20")
    assert keep_block("Plants make their own food. This is one way to tell plants apart.")


TOPICS = [
    "rivers", "rocks", "soil", "wind", "rain", "ice", "sand", "clay", "lava", "ash", "mud",
    "dust", "snow", "hail", "fog", "dew", "sun", "moon", "tide", "wave", "reef",
]


def test_boilerplate_lines_repeated_across_pages_are_removed():
    header = "Ilmu Pengetahuan Alam untuk SD Kelas IV"
    pages = [
        Page("book", i, [f"{header} {i}", f"A paragraph about {TOPICS[i]} and how it shapes the land."])
        for i in range(1, 21)
    ]
    cleaned = drop_boilerplate(pages)
    assert all(header not in b for p in cleaned for b in p.blocks)
    assert all(len(p.blocks) == 1 for p in cleaned)


def test_chunk_sizes_overlap_and_pages():
    words = [f"w{i}" for i in range(600)]
    pages = [Page("book", 1, [" ".join(words[:300])]), Page("book", 2, [" ".join(words[300:])])]
    chunks = chunk(pages, size=250, overlap=50)
    texts = [c["text"].split() for c in chunks]
    assert [len(t) for t in texts] == [250, 250, 200]
    assert texts[0][-50:] == texts[1][:50]  # overlap
    assert texts[-1][-1] == "w599"  # nothing lost at the end
    assert (chunks[0]["page_start"], chunks[0]["page_end"]) == (1, 1)
    assert (chunks[1]["page_start"], chunks[1]["page_end"]) == (1, 2)
    assert [c["id"] for c in chunks] == ["book:0", "book:1", "book:2"]


def test_chunk_skips_a_tail_that_is_only_overlap():
    pages = [Page("book", 1, [" ".join(f"w{i}" for i in range(260))])]
    assert len(chunk(pages, size=250, overlap=50)) == 2
    pages = [Page("book", 1, [" ".join(f"w{i}" for i in range(250))])]
    assert len(chunk(pages, size=250, overlap=50)) == 1


def test_wiki_pages_drop_section_headings(tmp_path):
    f = tmp_path / "Lever.txt"
    f.write_text(
        "A lever is a simple machine made of a beam and a fulcrum point.\n\n"
        "== History ==\nArchimedes described the lever and its law around 260 BC.\n"
    )
    (page,) = wiki_pages(f)
    assert page.source == "wiki:Lever"
    assert page.page is None
    assert all("==" not in b for b in page.blocks)
    assert len(page.blocks) == 2


def test_directive_blocks_are_dropped_in_both_languages():
    assert not keep_block("1. Apa saja benda langit yang ada di luar angkasa selain Bumi? "
                          "2. Diskusikan bersama kelompok kalian. 3. Tulislah jawabanmu di sini.")
    assert not keep_block("Analyze the graph and table to circle the answer. Explain your "
                          "reasoning. What do you notice about the pattern?")
    # Ordinary sentences whose first word only looks like a command stay.
    assert keep_block("Jumlah planet di tata surya ada delapan. Masalah utama adalah cahaya "
                      "matahari yang terhalang awan tebal.")
    assert keep_block("Work is done when a force moves an object over a distance. Use of "
                      "pulleys can reduce the force needed to lift a load.")


def test_reference_lists_and_worksheet_prompts_are_dropped():
    assert not keep_block("Cameron, Duncan. Soil Erosion and Degradation. Diakses 10 Maret 2024 "
                          "dari https://www.fao.org/soils")
    assert not keep_block("Observations & Wonderings What are you observing about this "
                          "phenomenon? What are you wondering about this phenomenon?")
    assert not keep_block("3.2.2 Analyze and interpret data to identify patterns of traits that "
                          "plants and animals have inherited from parents.")
    assert not keep_block("Terkait dimensi profil pelajar Pancasila yang kamu kembangkan selama "
                          "mempelajari bab ini, tuliskan pengalamanmu.")


def test_front_and_back_matter_pages_are_cut():
    from edge_grader.rag.corpus import body_pages

    text = "Sebuah paragraf panjang tentang energi dan cahaya matahari di bumi kita."
    pages = [Page("b", i, [text]) for i in range(1, 41)]
    pages[1].blocks.insert(0, "Kata Pengantar")
    pages[2].blocks.insert(0, "Daftar Isi")
    pages[35].blocks.insert(0, "Daftar Pustaka")
    kept = [p.page for p in body_pages(pages)]
    assert kept == list(range(4, 36))
    # A heading listed in the table of contents early on doesn't cut the book.
    pages[2].blocks.append("Glosarium")
    assert [p.page for p in body_pages(pages)] == list(range(4, 36))
