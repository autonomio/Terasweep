"""Exercise the offline learning page in Chromium and record a QA summary."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
checks = []


def check(name, result):
    assert result, name
    checks.append(name)


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("CHROMIUM_BIN"))
        page = browser.new_page(accept_downloads=True)
        errors, network = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: network.append(request.url))
        page.set_content((ROOT / "index.html").read_text(encoding="utf-8"), wait_until="load")
        page.wait_for_function("window.__MASTERCLASS_READY__ === true")
        chapters = page.evaluate("DATA.chapters.map(chapter => chapter.id)")
        for width in (320, 390, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            for chapter in chapters:
                page.evaluate("(id) => navigate(id)", chapter)
                check(f"{chapter} at {width}px", page.evaluate(
                    """() => document.querySelectorAll('.lesson.active').length === 1 &&
                        document.documentElement.scrollWidth <= innerWidth"""
                ))
                check(f"{chapter} readouts fit at {width}px", page.locator(
                    ".lesson.active .metric strong"
                ).evaluate_all("""nodes => nodes.every(node => {
                    const range = document.createRange();
                    range.selectNodeContents(node);
                    const box = node.getBoundingClientRect();
                    return [...range.getClientRects()].every(rect => rect.right <= box.right + 1);
                })"""))

        page.evaluate("navigate('model')")
        stats = page.locator("#ridge-metrics .metric").nth(1)
        gram = stats.inner_text()
        page.locator("#ridge-alpha").fill("2")
        page.locator("#ridge-alpha").dispatch_event("input")
        check("Ridge reuses the Gram statistic across regularization settings", stats.inner_text() == gram)
        page.locator("#ridge-intercept").uncheck()
        check("The intercept switch uses the raw statistic", stats.inner_text() != gram)

        page.evaluate("navigate('sweep')")
        cells = page.locator("#reuse-metrics strong")
        check("Four settings and 100 entries need four fits", cells.all_text_contents() == ["100", "4", "96"])
        page.locator("#fit-repeats").fill("1")
        page.locator("#fit-repeats").dispatch_event("input")
        check("Distinct entries all need fits", cells.all_text_contents() == ["4", "4", "0"])
        page.locator('[data-source="sweep"]').last.click()
        check("Evidence opens with current sweep code", "fn sweep" in page.locator("#source-text").inner_text())
        check("Evidence links to current package source", page.locator("#source-git").get_attribute("href").endswith("/main/src/sweep.rs"))
        page.keyboard.press("Escape")

        page.locator('[data-open="search"]').click()
        page.locator("#course-search").fill("paired")
        check("Search responds to unmatched queries", page.locator(".search-result").count() == 0)
        page.locator("#course-search").fill("upstream")
        page.locator('.search-result[data-go="reference"]').click()
        check("Search opens the chapter", page.locator("#reference.active").is_visible())
        page.locator('#reference [data-choice="1"]').click()
        check("Reference quiz gives feedback", page.locator(".quiz-feedback").inner_text().startswith("Yes."))
        page.locator("#mastered").click()
        check("Progress counts understood chapters", page.locator("#progress-count").inner_text() == f"1 / {len(chapters)}")
        page.locator("#lesson-note").fill("The validation split ranks a model.")
        page.evaluate("navigate('model'); navigate('reference')")
        check("Notes survive navigation", page.locator("#lesson-note").input_value() == "The validation split ranks a model.")
        with page.expect_download() as capture:
            page.locator("#export-notes-top").click()
        download = capture.value
        check("Export contains written notes", "The validation split ranks a model." in download.path().read_text())
        page.locator("#theme-btn").click()
        check("Theme changes", page.evaluate("document.documentElement.dataset.theme") == "light")

        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#menu-btn").click()
        page.locator('.nav-item[data-go="brief"]').click()
        check("Mobile drawer closes after navigation", page.locator("#menu-btn").get_attribute("aria-expanded") == "false")
        check("No browser errors", not errors)
        check("Works offline without network requests", not network)
        browser.close()

    report = {"status": "PASS", "chapters": len(chapters), "checks": checks}
    (ROOT / "qa-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(checks)} browser checks passed")


if __name__ == "__main__":
    main()
