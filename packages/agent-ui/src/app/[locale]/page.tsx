import ChatThread from '@/components/ChatThread';

export default async function Page({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = await params;
  // New conversation: agentserv lazily creates the row on the first message.
  return <ChatThread locale={locale} conversationId={null} />;
}
