def queryMfcTagsAndTitles(token: str, page: str) -> str:
    return """
{
  PremiumVideos(
    token: "%s"
    filters: [
      { key: "system_source_name", value: "mfc" }
      { key: "title", value: null, condition: GREATER_THAN}
      { key: "tags", value: null, condition: GREATER_THAN}
    ]
    pager: {
      size: 10000
      page: "%s"
    }
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
""" % (token, page)


def queryMfcTotal(token: str) -> str:
    return """
{
  PremiumVideos(
    token: "%s"
    filters: [
      { key: "system_source_name", value: "mfc" }
      { key: "title", value: null, condition: GREATER_THAN}
      { key: "tags", value: null, condition: GREATER_THAN}
    ]
  ) {
    Pager {
      total
    }
  }
}
""" % token
