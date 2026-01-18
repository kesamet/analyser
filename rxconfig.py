import reflex as rx

config = rx.Config(
    app_name="reflex_app",
    plugins=[
        rx.plugins.sitemap.SitemapPlugin()
    ]
)
