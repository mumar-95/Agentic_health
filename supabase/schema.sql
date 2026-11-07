-- Run this once in the Supabase SQL Editor (Dashboard → SQL → New query).
-- Safe to re-run: uses IF NOT EXISTS / OR REPLACE where possible.

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- Chat persistence
-- ---------------------------------------------------------------------------

create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.chat_sessions (id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  source text,
  created_at timestamptz not null default now()
);

create index if not exists chat_messages_session_id_idx
  on public.chat_messages (session_id, created_at);

create or replace function public.set_chat_session_updated_at()
returns trigger
language plpgsql
as $$
begin
  update public.chat_sessions
  set updated_at = now()
  where id = new.session_id;
  return new;
end;
$$;

drop trigger if exists chat_messages_touch_session on public.chat_messages;
create trigger chat_messages_touch_session
after insert on public.chat_messages
for each row
execute function public.set_chat_session_updated_at();

-- ---------------------------------------------------------------------------
-- Hospital document details
-- ---------------------------------------------------------------------------

create table if not exists public.hospital_documents (
  id uuid primary key default gen_random_uuid(),
  file_name text not null unique,
  file_type text not null,
  file_size_bytes bigint not null default 0,
  file_mtime timestamptz,
  content_text text not null default '',
  section_count integer not null default 0,
  chunk_count integer not null default 0,
  content_sha256 text,
  folder_fingerprint text,
  indexed_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists hospital_documents_file_name_idx
  on public.hospital_documents (file_name);

create index if not exists hospital_documents_indexed_at_idx
  on public.hospital_documents (indexed_at desc);

-- Optional per-chunk rows for inspection / auditing
create table if not exists public.hospital_document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.hospital_documents (id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index if not exists hospital_document_chunks_document_id_idx
  on public.hospital_document_chunks (document_id, chunk_index);

alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;
alter table public.hospital_documents enable row level security;
alter table public.hospital_document_chunks enable row level security;

-- Server uses SUPABASE_SECRET_KEY (bypasses RLS).
-- No public anon policies: data is not readable with the publishable key alone.
