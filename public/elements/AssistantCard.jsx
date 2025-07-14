import { Card, CardContent } from "@/components/ui/card";

export default function AssistantCard() {
  // @ignore: props
  if (!props) {
    console.error(
      "AssistantCard: props should be provided in parent component",
    );
    return null;
  }
  const content =
    props.content.replace(/\n/g, "<br />") || "No content provided";

  return (
    <div className="flex justify-start mb-4 relative">
      <Card id="assistant-card">
        <CardContent>
          <div
            style={{ whiteSpace: "pre-wrap" }}
            dangerouslySetInnerHTML={{ __html: content }}
          />
        </CardContent>
      </Card>
    </div>
  );
}
