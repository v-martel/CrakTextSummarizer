def queryMfcTagsAndTitles(token: str) -> str:
    return """
{
  PremiumVideos(
    token: "%s"
    filters: [
      { key: "system_source_name", value: "mfc" }
      { key: "tags", value: null, condition: GREATER_THAN}
    ]
  ) {
    Pager {
      size
      page
      nextPage
      total
    }
    Data {
      title
      tags
    }
  }
}
""" % token
